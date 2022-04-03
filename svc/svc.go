package svc

import (
	"fmt"
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
	Degree int64

	// Coef0 - свободный член для полиномиального ядра.
	Coef0 float64

	// Gamma - параметр для ядра 'rbf'.
	Gamma float64

	// Tol - точность.
	Tol float64

	// MaxIters - максимальное количество итераций.
	MaxIters int64

	// Индексы опорных векторов.
	supports []int64

	// Параметры для обучения алгоритма.
	nSamples  int64 // Число образцов
	nFeatures int64 // Число характеристик
	nClasses  int64 // Число классов

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
