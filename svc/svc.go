package svc

import (
	"fmt"
	"github.com/ziyadovea/svm/pkg/vector_operations"
	"strings"
)

// SVC (англ. Support Vector Classifier) - структура для представления
// классификатора методом опорных векторов.
type SVC struct {
	// Для линейно неразделимой выборки используется kernel trick.
	// kernelName - название ядра.
	kernelName KernelName

	// Kernel - ядро.
	Kernel Kernel

	// C - параметр регуляризации.
	C float64

	// Degree - степень многочлена для полиномиального ядра.
	Degree int

	// Coef0 - свободный член для полиномиального ядра.
	Coef0 float64

	// Gamma - параметр для ядра 'rbf'.
	Gamma float64

	// Tol - точность.
	Tol float64

	// MaxIters - максимальное количество итераций.
	MaxIters int

	// Индексы опорных векторов.
	supports []int

	// Параметры для обучения алгоритма.
	nSamples  int // Число образцов
	nFeatures int // Число характеристик
	nClasses  int // Число классов

	// Матрица признаков.
	x [][]float64

	// Метки классов.
	y []int

	// Параметры для решения QP методом SMO.
	b     float64
	k     float64
	alpha float64
}

// NewSVC возвращает экземпляр SVC с параметрами по умолчанию.
func NewSVC() *SVC {
	return &SVC{
		kernelName: "rbf",
		Kernel:     &RbfKernel{Gamma: 1.0},
		C:          1.0,
		Degree:     3,
		Coef0:      0.0,
		Gamma:      1.0,
		Tol:        0.001,
		MaxIters:   -1,
		supports:   nil,
		nSamples:   0,
		nFeatures:  0,
		nClasses:   0,
		x:          nil,
		y:          nil,
		b:          0.0,
		k:          0.0,
		alpha:      0.0,
	}
}

// SetKernelByName устанавливает ядро по его имени.
// Возвращает ошибку в случае неизвестного ядра.
func (svc *SVC) SetKernelByName(kernelName string) error {
	switch KernelName(strings.ToLower(kernelName)) {
	case LINEAR:
		svc.kernelName = LINEAR
		svc.Kernel = &LinearKernel{}
	case POLY:
		svc.kernelName = POLY
		svc.Kernel = &PolyKernel{
			Coef0:  svc.Coef0,
			Degree: svc.Degree,
		}
	case RBF:
		svc.kernelName = RBF
		svc.Kernel = &RbfKernel{Gamma: svc.Gamma}
	default:
		return fmt.Errorf("unknown kernel name")
	}
	return nil
}

// Fit обучает алгоритм на обучающей выборке.
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

	//

	return nil
}

// Predict классифицирует новые входные данные на основе обученной моодели.
func (svc *SVC) Predict(x [][]float64) ([]int, error) {
	return nil, nil
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

	return nil
}
