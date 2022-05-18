package svc

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/jinzhu/copier"
	"github.com/ziyadovea/svm"
	"github.com/ziyadovea/svm/pkg/vector_operations"
)

// Проверим, что структура SVC удовлетворяет интерфейсу Classifier.
var _ svm.Classifier = (*SVC)(nil)

// SVC (англ. Support Vector Classifier) - структура для представления
// классификатора методом опорных векторов.
type SVC struct {
	// Для линейно неразделимой выборки используется kernel trick.
	// Название ядра.
	kernelName KernelName

	// Ядро.
	Kernel Kernel

	// Кэш ядра - матрица скалярных произведений.
	kernelCache [][]float64

	// Параметр регуляризации.
	C float64

	// Степень многочлена для полиномиального ядра.
	Degree int

	// Свободный член для полиномиального ядра.
	Coef0 float64

	// Параметр для ядра 'rbf'.
	Gamma float64

	// Точность.
	Tol float64

	// Максимальное количество итераций.
	MaxIters int

	// Вектор с индексами опорных векторов.
	supportVectorsIdx []int

	// Матрица признаков обучающей выборки.
	x [][]float64

	// Метки классов обучающей выборки.
	y []int

	// Порог для SVM.
	b float64

	// Число образцов обучающей выборки
	nSamples int
	// Число характеристик обучающей выборки
	nFeatures int
	// Число классов обучающей выборки
	nClasses int

	// Параметры для решения QP методом SMO.
	alphas []float64 // Альфа-параметры опорных векторов.
}

// NewSVC возвращает экземпляр SVC с параметрами по умолчанию.
func NewSVC() *SVC {
	return &SVC{
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
	}
}

// SetKernelByName устанавливает ядро по его имени.
// Возвращает ошибку в случае неизвестного ядра.
func (svc *SVC) SetKernelByName(kernelName string) error {
	switch KernelName(strings.ToLower(kernelName)) {
	case Linear:
		svc.kernelName = Linear
		svc.Kernel = &LinearKernel{}
	case Poly:
		svc.kernelName = Poly
		svc.Kernel = &PolyKernel{
			Coef0:  svc.Coef0,
			Degree: svc.Degree,
		}
	case Rbf:
		svc.kernelName = Rbf
		svc.Kernel = &RbfKernel{Gamma: svc.Gamma}
	default:
		return fmt.Errorf("unknown kernel name")
	}
	return nil
}

// Fit обучает алгоритм на обучающей выборке.
// x - матрица признаков.
// y - слайс меток, y = +1 или -1.
func (svc *SVC) Fit(x [][]float64, y []int) error {
	// Проверим валидность входных данных.
	if err := svc.validateInput(x, y); err != nil {
		return fmt.Errorf("invalid input data: %w", err)
	}

	// Запишем в поля структуры необходимые данные.
	svc.nSamples = len(x)
	svc.nFeatures = len(x[0])
	svc.x = x
	svc.y = y

	// Закэшируем произведения ядра.
	svc.cacheKernel()

	// Сначала опорными являются все вектора.
	svc.supportVectorsIdx = make([]int, svc.nSamples)
	for i := 0; i < svc.nSamples; i++ {
		svc.supportVectorsIdx[i] = i
	}

	// Начнем обучение алгоритма.
	// Для обучения алгоритма необходимо решение задачи QP.
	// Воспользуемся популярным и эффективным методом решения этой задачи - SMO.
	log.Println("SVM fitting started...")
	svc.smo()

	return nil
}

// Predict классифицирует новые входные данные на основе обученной моодели.
// x - матрица признаков.
func (svc *SVC) Predict(x [][]float64) []int {
	labels := make([]int, len(x))
	for i := range x {
		value := svc.f(x[i])
		if value >= 0 {
			labels[i] = 1
		} else {
			labels[i] = -1
		}
	}
	return labels
}

// Вычисления f(x).
func (svc *SVC) f(x []float64) float64 {
	result := 0.0
	for i := range svc.supportVectorsIdx {
		result += svc.alphas[i] * float64(svc.y[i]) * svc.Kernel.Calculate(svc.x[i], x)
	}
	return result + svc.b
}

// validateInput проверяет валидность входных данных для обучения - массива меток и матрицы признаков.
func (svc *SVC) validateInput(x [][]float64, y []int) error {
	// Базовый SVM является бинарным - он работает только с 2 классами.
	// Если в исходной разметке классов больше - надо выдать ошибку.
	svc.nClasses = vector_operations.CountOfUniques(y)
	if svc.nClasses != 2 {
		return fmt.Errorf("incorrect number of class labels: expected 2, actual: %d", svc.nClasses)
	}

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

// smo представляет реализацию метода SMO для решения задачи QP.
func (svc *SVC) smo() {
	log.Println("Started solving the QP problem by the SMO")

	// Изначально alphas - массив размером nSamples из нулей.
	svc.alphas = make([]float64, svc.nSamples)

	iterCounter := 0
	// Главный цикл. Он завершится раньше, если решение сойдется меньше, чем за
	// максимальное количество итераций.
	for iterCounter < svc.MaxIters {
		log.Printf("%d SMO iteration out of %d\n", iterCounter, svc.MaxIters)

		// Количество измененных параметров альфа за текущую итерацию
		numChangedAlphas := 0
		for i := 0; i < svc.nSamples; i++ {
			// Ошибка для i-ого экземпляра
			errI := svc.f(svc.x[i]) - float64(svc.y[i])

			// Проверяем выполнение условий ККТ
			if (float64(svc.y[i])*errI < -svc.Tol && svc.alphas[i] < svc.C) ||
				(float64(svc.y[i])*errI > svc.Tol && svc.alphas[i] > 0) {

				// Выбираем рандомный индекс j != i.
				// Это является упрощенной эвристикой для выбора оптимальных параметров alpha[i] и alpha[j].
				j := svc.getJ(i)

				// Ошибка для j-ого экземпляра
				errJ := svc.f(svc.x[j]) - float64(svc.y[j])

				// Сохраняем старые значения параметров альфа
				alphaIOld := svc.alphas[i]
				alphaJOld := svc.alphas[j]

				// Считаем границы L и H
				L := 0.0
				H := 0.0
				if svc.y[i] != svc.y[j] {
					L = math.Max(0, svc.alphas[j]-svc.alphas[i])
					H = math.Min(svc.C, svc.C+svc.alphas[j]-svc.alphas[i])
				} else {
					L = math.Max(0, svc.alphas[j]+svc.alphas[i]-svc.C)
					H = math.Min(svc.C, svc.alphas[j]+svc.alphas[i])
				}

				// Если границы равны, то переход к след. итерации
				if L == H {
					continue
				}

				// Считаем параметр
				eta := 2*svc.kernelCache[i][j] - svc.kernelCache[i][i] - svc.kernelCache[j][j]

				// Если значение >=0, то переход к след. итерации
				if eta >= 0 {
					continue
				}

				// Обновляем значения параметров альфа
				svc.alphas[j] -= (float64(svc.y[j]) * (errI - errJ)) / eta
				if svc.alphas[j] > H {
					svc.alphas[j] = H
				} else if svc.alphas[j] < L {
					svc.alphas[j] = L
				}

				// Проверим разность параметров альфа
				if math.Abs(svc.alphas[j]-alphaJOld) < svc.Tol {
					continue
				}

				// Определим значение параметра alpha[i]
				svc.alphas[i] += float64(svc.y[i]) * float64(svc.y[j]) * (alphaJOld - svc.alphas[j])

				// Находим значение параметра b
				b1 := svc.b - errI - float64(svc.y[i])*(svc.alphas[i]-alphaIOld)*svc.kernelCache[i][j] -
					float64(svc.y[j])*(svc.alphas[j]-alphaJOld)*svc.kernelCache[i][j]
				b2 := svc.b - errJ - float64(svc.y[i])*(svc.alphas[i]-alphaIOld)*svc.kernelCache[i][j] -
					float64(svc.y[j])*(svc.alphas[j]-alphaJOld)*svc.kernelCache[j][i]

				if 0 < svc.alphas[i] && svc.alphas[i] < svc.C {
					svc.b = b1
				} else if 0 < svc.alphas[j] && svc.alphas[j] < svc.C {
					svc.b = b2
				} else {
					svc.b = (b1 + b2) / 2
				}

				// Изменим количество измененных значений альфа
				numChangedAlphas++
			}
		}

		if numChangedAlphas == 0 {
			break
		}

		iterCounter++
	}

	// Теперь надо сохранить индексы опорных векторов
	svc.supportVectorsIdx = make([]int, 0)
	for i := 0; i < svc.nSamples; i++ {
		if svc.alphas[i] > svc.Tol {
			svc.supportVectorsIdx = append(svc.supportVectorsIdx, i)
		}
	}
}

// Возвращает случайное значение в диапазоне [0, svc.nSamples - 1], не равное i.
func (svc *SVC) getJ(i int) int {
	rand.Seed(time.Now().UnixNano())
	res := rand.Intn(svc.nSamples)
	for res == i {
		res = rand.Intn(svc.nSamples)
	}
	return res
}

// Кэшируем значения скалярных произведений ядра,
// чтобы брать значения из кэша, а не считать на каждой итерации.
func (svc *SVC) cacheKernel() {
	// Инициализация матрицы.
	svc.kernelCache = make([][]float64, svc.nSamples)
	for i := 0; i < svc.nSamples; i++ {
		svc.kernelCache[i] = make([]float64, svc.nSamples)
	}

	// Заполнение значениями.
	for i := 0; i < svc.nSamples; i++ {
		for j := 0; j < svc.nSamples; j++ {
			svc.kernelCache[i][j] = svc.Kernel.Calculate(svc.x[i], svc.x[j])
		}
	}
}

// Clone возвращает копию SVM.
func (svc *SVC) Clone() (svm.Classifier, error) {
	res := &SVC{}
	if err := copier.Copy(res, svc); err != nil {
		return nil, err
	}
	return res, nil
}
