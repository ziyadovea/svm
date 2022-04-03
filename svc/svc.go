package svc

// SVC (англ. Support Vector Classifier) - структура для представления
// классификатора методом опорных векторов.
type SVC struct {
	// Для линейно неразделимой выборки используется kernel trick.
	// KernelName - название ядра.
	KernelName string

	// Kernel - ядро.
	kernel Kernel

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

	// MaxIters - максимальное количество итераций при
	MaxIters int64

	// Индексы опорных векторов.
	Supports []int

	// Параметры для решения QP методом SMO
	b     float64
	k     float64
	alpha float64
}

// NewSVC возвращает экземпляр SVC с параметрами по умолчанию
func NewSVC() *SVC {
	return &SVC{}
}
