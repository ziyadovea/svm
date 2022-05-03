package svc

import (
	"github.com/ziyadovea/svm/pkg/vector_operations"
	"math"
)

// KernelName тип для имени ядра.
type KernelName string

// Определяем в константах существующие ядра.
const (
	Linear KernelName = "linear"
	Poly   KernelName = "poly"
	Rbf    KernelName = "rbf"
)

// Kernel интерфейс для ядра.
type Kernel interface {
	Calculate(x, y []float64) float64
}

// Удостоверяемся, что структура LinearKernel удовлетворяет интерфейсу Kernel.
var _ Kernel = (*LinearKernel)(nil)

// LinearKernel представляет линейное ядро.
type LinearKernel struct{}

// Calculate cчитает скалярное произведение двух векторов.
func (k *LinearKernel) Calculate(x, y []float64) float64 {
	return vector_operations.ScalarProduct(x, y)
}

// Удостоверяемся, что структура PolyKernel удовлетворяет интерфейсу Kernel.
var _ Kernel = (*PolyKernel)(nil)

// PolyKernel представляет собой полиномиальное ядро.
type PolyKernel struct {
	Coef0  float64
	Degree int
}

// Calculate считает произведение двух векторов
// по формуле z = (<x, y> + r)^d, где:
// 1. <x, y> - скалярное произведение векторов;
// 2. r - свободный член;
// 3. d - степень полинома.
func (k *PolyKernel) Calculate(x, y []float64) float64 {
	scalarProduct := vector_operations.ScalarProduct(x, y)
	return math.Pow(scalarProduct+k.Coef0, float64(k.Degree))
}

// Удостоверяемся, что структура RbfKernel удовлетворяет интерфейсу Kernel.
var _ Kernel = (*RbfKernel)(nil)

// RbfKernel представляет собой ядро rbf - radial basic function.
type RbfKernel struct {
	Gamma float64
}

// Calculate считает произведение двух векторов
// по формуле z = exp{-gamma * |x - y|^2}, gamma > 0.
func (k *RbfKernel) Calculate(x, y []float64) float64 {
	euclideanDistance := vector_operations.EuclideanDistance(x, y)
	return math.Exp(-k.Gamma * math.Pow(euclideanDistance, 2))
}
