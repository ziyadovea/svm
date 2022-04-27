package binary_metrics

import (
	"fmt"
)

// GetConfusionMatrix вычисляет матрицу ошибок.
// Возвращает ее в 2 видах: в виде матрицы 2 на 2
// и в виде строки для красивого вывода.
func GetConfusionMatrix(yTrue []int, yPred []int) ([2][2]int, string) {
	tp, fp, fn, tn := 0, 0, 0, 0
	for i := range yTrue {
		if yTrue[i] == +1 && yPred[i] == +1 {
			tp++
		}
		if yTrue[i] == -1 && yPred[i] == +1 {
			fp++
		}
		if yTrue[i] == +1 && yPred[i] == -1 {
			fn++
		}
		if yTrue[i] == -1 && yPred[i] == -1 {
			tn++
		}
	}

	confusionMatrixStr := fmt.Sprintf(`
Confusion matrix
 -------
| %d | %d |
 -------
| %d | %d |
 -------
`, tn, fp, fn, tp)

	return [2][2]int{
		{tn, fp},
		{fn, tp},
	}, confusionMatrixStr
}

// Accuracy вычисляет метрику классификации Accuracy.
func Accuracy(yTrue []int, yPred []int) float64 {
	countOfCorrectAnswers := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			countOfCorrectAnswers++
		}
	}
	return float64(countOfCorrectAnswers) / float64(len(yTrue))
}

// Precision вычисляет метрику классификации Precision.
func Precision(yTrue []int, yPred []int) float64 {
	cm, _ := GetConfusionMatrix(yTrue, yPred)
	return float64(cm[1][1]) / float64(cm[1][1]+cm[0][1])
}

// Recall вычисляет метрику классификации Recall.
func Recall(yTrue []int, yPred []int) float64 {
	cm, _ := GetConfusionMatrix(yTrue, yPred)
	return float64(cm[1][1]) / float64(cm[1][1]+cm[1][0])
}

// FScore вычисляет метрику классификации F-мера.
func FScore(yTrue []int, yPred []int) float64 {
	precision := Precision(yTrue, yPred)
	recall := Recall(yTrue, yPred)
	return 2 * precision * recall / (precision + recall)
}

// ExtendedFScore вычисляет метрику классификации расширенная F-мера.
func ExtendedFScore(yTrue []int, yPred []int, beta float64) float64 {
	precision := Precision(yTrue, yPred)
	recall := Recall(yTrue, yPred)
	return (1 + beta*beta) * precision * recall / (beta*beta*precision + recall)
}
