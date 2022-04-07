package vector_operations

import "math"

// ScalarProduct считает скалярное произведение двух векторов
// по формуле z = <x, y> = x1*y1 + x2*y2 + ... + xn*yn.
func ScalarProduct(x, y []float64) float64 {
	var result float64
	for i := range x {
		result += x[i] * y[i]
	}
	return result
}

// EuclideanDistance считает евклидово расстояние между двумя векторами
// по формуле z = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2).
func EuclideanDistance(x, y []float64) float64 {
	var result float64
	for i := range x {
		result += math.Pow(x[i]-y[i], 2)
	}
	return math.Sqrt(result)
}

// CountOfUniques считает количество уникальных элементов в слайсе.
func CountOfUniques(x []int) int {
	set := make(map[int]struct{}, len(x))
	for i := range x {
		set[x[i]] = struct{}{}
	}
	return len(set)
}

// IsMatrixRectangular проверяет, что матрица является прямоугольной.
func IsMatrixRectangular(x [][]float64) bool {
	len0 := len(x[0])
	for i := 1; i < len(x); i++ {
		if len(x[i]) != len0 {
			return false
		}
	}
	return true
}
