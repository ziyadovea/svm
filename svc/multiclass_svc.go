package svc

import (
	"fmt"
	"strings"
	"sync"

	"github.com/jinzhu/copier"
	"github.com/ziyadovea/svm"
	"github.com/ziyadovea/svm/pkg/vector_operations"
	"golang.org/x/sync/errgroup"
)

// Проверим, что структура MultiSVC удовлетворяет интерфейсу Classifier.
var _ svm.Classifier = (*MultiSVC)(nil)

// MultiSVC (англ. Multiclass Support Vector Classifier) - структура для представления
// мультиклассового классификатора методом опорных векторов.
// Реализуется метод One-vs-All.
type MultiSVC struct {
	SVC

	// Карта SVM-ов для каждого бинарного случая.
	// Ключ - метка класса, значение - классификатор.
	Machines map[int]*SVC

	// Слайс уникальных меток обучающего набора.
	labels []int
}

// NewMultiSVC возвращает экземпляр MultiSVC с параметрами по умолчанию.
func NewMultiSVC() *MultiSVC {
	return &MultiSVC{
		SVC: SVC{
			kernelName:        "rbf",
			Kernel:            &RbfKernel{Gamma: 1.0},
			kernelCache:       nil,
			C:                 1.0,
			Degree:            3,
			Coef0:             0.0,
			Gamma:             1.0,
			Tol:               0.001,
			MaxIters:          10000,
			supportVectorsIdx: nil,
			x:                 nil,
			y:                 nil,
			b:                 0.0,
			nSamples:          0,
			nFeatures:         0,
			nClasses:          0,
			alphas:            nil,
		},
		Machines: nil,
		labels:   nil,
	}
}

// SetKernelByName устанавливает ядро по его имени.
// Возвращает ошибку в случае неизвестного ядра.
func (m *MultiSVC) SetKernelByName(kernelName string) error {
	switch KernelName(strings.ToLower(kernelName)) {
	case Linear:
		m.kernelName = Linear
		m.Kernel = &LinearKernel{}
	case Poly:
		m.kernelName = Poly
		m.Kernel = &PolyKernel{
			Coef0:  m.Coef0,
			Degree: m.Degree,
		}
	case Rbf:
		m.kernelName = Rbf
		m.Kernel = &RbfKernel{Gamma: m.Gamma}
	default:
		return fmt.Errorf("unknown kernel name")
	}
	return nil
}

// Fit обучает алгоритм на обучающей выборке.
// x - матрица признаков.
// y - слайс меток.
func (m *MultiSVC) Fit(x [][]float64, y []int) error {
	// Проверим валидность входных данных.
	if err := m.validateInput(x, y); err != nil {
		return fmt.Errorf("invalid input data: %w", err)
	}

	// Выделим память под все бинарные классификаторы. Их число равно количеству классов.
	m.labels = vector_operations.GetUniques(y)
	m.nClasses = len(m.labels)
	m.Machines = make(map[int]*SVC, m.nClasses)

	// Создаем errgroup.Group для обучения каждого бинарного классификатора в отдельной горутине.
	eg := new(errgroup.Group)
	// Создаем мьютекс для добавления элементов в мапу, так как мапа в Go потоко-небезопасный тип.
	mu := sync.Mutex{}

	// Классификация методом One-vs-All
	for _, label := range m.labels {
		label := label
		eg.Go(func() error {
			// Помечаем текущий класс как +1, все остальные - как -1
			yTmp := make([]int, len(y))
			for i := range y {
				if y[i] == label {
					yTmp[i] = +1
				} else {
					yTmp[i] = -1
				}
			}

			// Создаем очередной бинарный классификатор
			svc := NewSVC()
			svc.kernelName = m.kernelName
			svc.Kernel = m.Kernel
			svc.C = m.C
			svc.Degree = m.Degree
			svc.Coef0 = m.Coef0
			svc.Gamma = m.Gamma
			svc.Tol = m.Tol
			svc.MaxIters = m.MaxIters

			// Обучаем очередной бинарный классификатор
			if err := svc.Fit(x, yTmp); err != nil {
				return fmt.Errorf("error in fitting a binary classifier: %w", err)
			}

			// Добавляем в слайс машин уже обученный экземпляр SVC
			mu.Lock()
			m.Machines[label] = svc
			mu.Unlock()

			return nil
		})
	}

	if err := eg.Wait(); err != nil {
		return err
	}

	return nil
}

// validateInput проверяет валидность входных данных для обучения - массива меток и матрицы признаков.
func (m *MultiSVC) validateInput(x [][]float64, y []int) error {
	// Проверим, что матрица признаков является прямоугольной.
	if !vector_operations.IsMatrixRectangular(x) {
		return fmt.Errorf("feature matrix must be rectangular")
	}

	// Проверим, что все данные размечены
	if len(x) != len(y) {
		return fmt.Errorf("not all data is labeled")
	}

	return nil
}

// Predict классифицирует новые входные данные на основе обученной моодели.
// x - матрица признаков.
func (m *MultiSVC) Predict(x [][]float64) []int {
	res := make([]int, len(x))
	for i := range x {
		res[i] = m.predictOne(x[i])
	}
	return res
}

// Возвращает метку класса, к которой обученный классификатор отнес объект с признаковым описанием x.
func (m *MultiSVC) predictOne(x []float64) int {
	results := make(map[int]float64, len(m.labels))

	// Начинем классификацию с самого частого класса
	for _, label := range m.labels {
		results[label] = m.Machines[label].f(x)
	}

	// Найдем класс с наиболее вероятным результатом
	pl := vector_operations.SortByValue(results)
	return pl[0].Key
}

// Clone возвращает копию мультиклассового SVM.
func (m *MultiSVC) Clone() (svm.Classifier, error) {
	res := &MultiSVC{}
	if err := copier.Copy(res, m); err != nil {
		return nil, err
	}
	return res, nil
}
