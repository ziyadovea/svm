package multiclass_metrics

import "github.com/ziyadovea/svm/pkg/vector_operations"

// GetConfusionMatrix вычисляет матрицу ошибок
// для случая мультиклассификации.
func GetConfusionMatrix(yTrue []int, yPred []int) [][]int {
	classesCount := vector_operations.CountOfUniques(yTrue)
	cm := make([][]int, classesCount)
	for i := 0; i < classesCount; i++ {
		cm[i] = make([]int, classesCount)
	}

	classes := vector_operations.GetUniques(yTrue)
	for i := 0; i < classesCount; i++ {
		for j := 0; j < classesCount; j++ {
			// считаем очередной элемент матрицы ошибок
			elem := 0
			for index := range yTrue {
				if yTrue[index] == classes[i] && yPred[index] == classes[j] {
					elem++
				}
			}
			cm[i][j] = elem
		}
	}

	return cm
}
