package vector_operations

import (
	"math"
	"sort"
)

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
	return len(GetUniques(x))
}

// GetUniques возвращает уникальные элементы слайса x
// в отсортированном порядке.
func GetUniques(x []int) []int {
	set := make(map[int]struct{}, 0)
	for i := range x {
		set[x[i]] = struct{}{}
	}

	res := make([]int, 0, len(set))
	for k, _ := range set {
		res = append(res, k)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i] < res[j]
	})

	return res
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

// Counter возвращает мапу вида "элемент : количество таких элементов в слайсе".
func Counter(x []int) map[int]int {
	uniqs := GetUniques(x)
	res := make(map[int]int, len(uniqs))
	for _, uniq := range uniqs {
		res[uniq] = Count(x, uniq)
	}
	return res
}

// Count возвращает число вхождений элемента el в слайс x.
func Count(x []int, el int) int {
	res := 0
	for _, xx := range x {
		if xx == el {
			res++
		}
	}
	return res
}

// Average возвращает среднее значение слайса x.
func Average(x []float64) float64 {
	res := 0.0
	for _, item := range x {
		res += item
	}
	return res / float64(len(x))
}

// IsBinary возвращает true, если x является бинарным (состоит только из 0 и 1),
// иначе - false.
func IsBinary(x []int) bool {
	for _, item := range x {
		if item != 0 && item != 1 {
			return false
		}
	}
	return true
}

// SortByValue сортирует мапу по значению по убыванию.
func SortByValue(m map[int]int) PairList {
	pl := make(PairList, len(m))
	i := 0
	for k, v := range m {
		pl[i] = Pair{k, v}
		i++
	}
	sort.Sort(sort.Reverse(pl))
	return pl
}

type Pair struct {
	Key   int
	Value int
}

type PairList []Pair

func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value < p[j].Value }
func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
