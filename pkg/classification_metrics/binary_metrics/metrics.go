package binary_metrics

import (
	"fmt"
	"math"
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
	res := float64(cm[1][1]) / float64(cm[1][1]+cm[0][1])
	if math.IsNaN(res) {
		return 0.0
	}
	return res
}

// Recall вычисляет метрику классификации Recall.
func Recall(yTrue []int, yPred []int) float64 {
	cm, _ := GetConfusionMatrix(yTrue, yPred)
	res := float64(cm[1][1]) / float64(cm[1][1]+cm[1][0])
	if math.IsNaN(res) {
		return 0.0
	}
	return res
}

// FScore вычисляет метрику классификации F-мера.
func FScore(yTrue []int, yPred []int) float64 {
	precision := Precision(yTrue, yPred)
	recall := Recall(yTrue, yPred)
	res := 2 * precision * recall / (precision + recall)
	if math.IsNaN(res) {
		return 0.0
	}
	return res
}

// FBetaScore вычисляет метрику классификации расширенная F-мера.
func FBetaScore(yTrue []int, yPred []int, beta float64) float64 {
	precision := Precision(yTrue, yPred)
	recall := Recall(yTrue, yPred)
	res := (1 + beta*beta) * precision * recall / (beta*beta*precision + recall)
	if math.IsNaN(res) {
		return 0.0
	}
	return res
}

// BinaryClassificationReport вовзращает отчет по метрикам бинарной классификации.
func BinaryClassificationReport(yTrue []int, yPred []int) (cm [2][2]int, accuracy, precision, recall, f1 float64, reportString string) {
	cm, reportString = GetConfusionMatrix(yTrue, yPred)
	accuracy = Accuracy(yTrue, yPred)
	precision = Precision(yTrue, yPred)
	recall = Recall(yTrue, yPred)
	f1 = FScore(yTrue, yPred)
	reportString += fmt.Sprintf(`
Accuracy  = %.3f
Precision = %.3f
Recall    = %.3f
F-score   = %.3f
`, accuracy, precision, recall, f1)
	return
}
