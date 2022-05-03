package multiclass_metrics

import (
	"fmt"
	"github.com/ziyadovea/svm/pkg/classification_metrics/binary_metrics"
	"github.com/ziyadovea/svm/pkg/vector_operations"
	"math"
	"strings"
)

// Average тип для вида метрик многоклассовой классификации
type Average string

const (
	Macro    Average = "macro"
	Weighted Average = "weighted"
	Micro    Average = "micro"
)

// GetConfusionMatrix вычисляет матрицу ошибок
// для случая мультиклассификации.
func GetConfusionMatrix(yTrue []int, yPred []int) ([][]int, string) {
	classesCount := vector_operations.CountOfUniques(yTrue)
	cm := make([][]int, classesCount)
	for i := 0; i < classesCount; i++ {
		cm[i] = make([]int, classesCount)
	}

	// заводим билдер строки для красивого вывода матрицы ошибок
	sb := strings.Builder{}
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

			// записываем очередной элемент в билдер
			if j != classesCount-1 {
				sb.WriteString(fmt.Sprintf("%d ", elem))
			} else {
				sb.WriteString(fmt.Sprintf("%d", elem))
			}
		}

		// перевод на новую строку
		if i != classesCount-1 {
			sb.WriteString("\n")
		}
	}

	cmStr := fmt.Sprintf(`
Confusion matrix
%s
`, sb.String())

	return cm, cmStr
}

// Accuracy вычисляет метрику классификации Accuracy.
func Accuracy(yTrue []int, yPred []int) float64 {
	return binary_metrics.Accuracy(yTrue, yPred)
}

// Precision вычисляет метрику классификации Precision.
func Precision(yTrue []int, yPred []int, average Average) float64 {
	res := 0.0

	// Для микро-усреднения
	tpMicro, fpMicro := 0.0, 0.0

	// Для взвешенного усреднения
	classCount := vector_operations.Counter(yTrue)

	// Сначала нам надо найти метрики отдельных задач классификации
	classes := vector_operations.GetUniques(yTrue)
	classesCount := len(classes)
	for _, class := range classes {
		// Находим массивы настоящих и предсказанных метрик классов для очередного класса
		tmpTrue := make([]int, len(yTrue))
		tmpPred := make([]int, len(yPred))
		for i := range yTrue {
			if yTrue[i] == class {
				tmpTrue[i] = +1
			} else {
				tmpTrue[i] = -1
			}

			if yPred[i] == class {
				tmpPred[i] = +1
			} else {
				tmpPred[i] = -1
			}
		}

		// Затем находим матрицу ошибок для текущего класса
		cm, _ := binary_metrics.GetConfusionMatrix(tmpTrue, tmpPred)
		tp, fp := cm[1][1], cm[0][1]

		// Затем в зависимости от параметра average надо это метрики усреднить
		switch average {
		case Macro:
			tmpPrecision := float64(tp) / float64(tp+fp)
			res += tmpPrecision
		case Weighted:
			tmpPrecision := float64(tp) / float64(tp+fp)
			weightedPrecision := float64(classCount[class]) * tmpPrecision
			res += weightedPrecision
		case Micro:
			tpMicro += float64(tp)
			fpMicro += float64(fp)
		}
	}

	// В конце также надо усреднить ответ в зависимости от параметра
	switch average {
	case Macro:
		res /= float64(classesCount)
	case Weighted:
		res /= float64(len(yTrue))
	case Micro:
		res = tpMicro / (tpMicro + fpMicro)
	}

	if math.IsNaN(res) {
		return 0.0
	}

	return res
}

// Recall вычисляет метрику классификации Recall.
func Recall(yTrue []int, yPred []int, average Average) float64 {
	res := 0.0

	// Для микро-усреднения
	tpMicro, fnMicro := 0.0, 0.0

	// Для взвешенного усреднения
	classCount := vector_operations.Counter(yTrue)

	// Сначала нам надо найти метрики отдельных задач классификации
	classes := vector_operations.GetUniques(yTrue)
	classesCount := len(classes)
	for _, class := range classes {
		// Находим массивы настоящих и предсказанных метрик классов для очередного класса
		tmpTrue := make([]int, len(yTrue))
		tmpPred := make([]int, len(yPred))
		for i := range yTrue {
			if yTrue[i] == class {
				tmpTrue[i] = +1
			} else {
				tmpTrue[i] = -1
			}

			if yPred[i] == class {
				tmpPred[i] = +1
			} else {
				tmpPred[i] = -1
			}
		}

		// Затем находим матрицу ошибок для текущего класса
		cm, _ := binary_metrics.GetConfusionMatrix(tmpTrue, tmpPred)
		tp, fn := cm[1][1], cm[1][0]

		// Затем в зависимости от параметра average надо это метрики усреднить
		switch average {
		case Macro:
			tmpRecall := float64(tp) / float64(tp+fn)
			res += tmpRecall
		case Weighted:
			tmpRecall := float64(tp) / float64(tp+fn)
			weightedRecall := float64(classCount[class]) * tmpRecall
			res += weightedRecall
		case Micro:
			tpMicro += float64(tp)
			fnMicro += float64(fn)
		}
	}

	// В конце также надо усреднить ответ в зависимости от параметра
	switch average {
	case Macro:
		res /= float64(classesCount)
	case Weighted:
		res /= float64(len(yTrue))
	case Micro:
		res = tpMicro / (tpMicro + fnMicro)
	}

	if math.IsNaN(res) {
		return 0.0
	}

	return res
}

// FScore вычисляет метрику классификации F-мера.
func FScore(yTrue []int, yPred []int, average Average) float64 {
	// В случае микроусреднения достаточно посчитать Precision и Recall
	// а затем посчитать для них F1-score
	if average == Micro {
		precision := Precision(yTrue, yPred, Micro)
		recall := Recall(yTrue, yPred, Micro)
		res := 2 * precision * recall / (precision + recall)
		if math.IsNaN(res) {
			return 0.0
		}
		return res
	}

	// Для макроусреднения и взвешеннного усреднения же требуется
	// более сложная логика, подобная подсчету предыдущих метрик
	res := 0.0

	// Для взвешенного усреднения
	classCount := vector_operations.Counter(yTrue)

	// Сначала нам надо найти метрики отдельных задач классификации
	classes := vector_operations.GetUniques(yTrue)
	classesCount := len(classes)
	for _, class := range classes {
		// Находим массивы настоящих и предсказанных метрик классов для очередного класса
		tmpTrue := make([]int, len(yTrue))
		tmpPred := make([]int, len(yPred))
		for i := range yTrue {
			if yTrue[i] == class {
				tmpTrue[i] = +1
			} else {
				tmpTrue[i] = -1
			}

			if yPred[i] == class {
				tmpPred[i] = +1
			} else {
				tmpPred[i] = -1
			}
		}

		// Затем находим матрицу ошибок для текущего класса
		cm, _ := binary_metrics.GetConfusionMatrix(tmpTrue, tmpPred)
		tp, fn, fp := cm[1][1], cm[1][0], cm[0][1]

		// Затем в зависимости от параметра average надо это метрики усреднить
		switch average {
		case Macro:
			tmpRecall := float64(tp) / float64(tp+fn)
			tmpPrecision := float64(tp) / float64(tp+fp)
			tmpFScore := 2 * tmpPrecision * tmpRecall / (tmpPrecision + tmpRecall)
			res += tmpFScore
		case Weighted:
			tmpRecall := float64(tp) / float64(tp+fn)
			tmpPrecision := float64(tp) / float64(tp+fp)
			tmpFScore := 2 * tmpPrecision * tmpRecall / (tmpPrecision + tmpRecall)
			weightedFScore := float64(classCount[class]) * tmpFScore
			res += weightedFScore
		}
	}

	// В конце также надо усреднить ответ в зависимости от параметра
	switch average {
	case Macro:
		res /= float64(classesCount)
	case Weighted:
		res /= float64(len(yTrue))
	}

	if math.IsNaN(res) {
		return 0.0
	}

	return res
}

// FBetaScore вычисляет метрику классификации расширенная F-мера.
func FBetaScore(yTrue []int, yPred []int, beta float64, average Average) float64 {
	// В случае микроусреднения достаточно посчитать Precision и Recall
	// а затем посчитать для них F beta score
	if average == Micro {
		precision := Precision(yTrue, yPred, Micro)
		recall := Recall(yTrue, yPred, Micro)
		return (1 + beta*beta) * precision * recall / (beta*beta*precision + recall)
	}

	// Для макроусреднения и взвешеннного усреднения же требуется
	// более сложная логика, подобная подсчету предыдущих метрик
	res := 0.0

	// Для взвешенного усреднения
	classCount := vector_operations.Counter(yTrue)

	// Сначала нам надо найти метрики отдельных задач классификации
	classes := vector_operations.GetUniques(yTrue)
	classesCount := len(classes)
	for _, class := range classes {
		// Находим массивы настоящих и предсказанных метрик классов для очередного класса
		tmpTrue := make([]int, len(yTrue))
		tmpPred := make([]int, len(yPred))
		for i := range yTrue {
			if yTrue[i] == class {
				tmpTrue[i] = +1
			} else {
				tmpTrue[i] = -1
			}

			if yPred[i] == class {
				tmpPred[i] = +1
			} else {
				tmpPred[i] = -1
			}
		}

		// Затем находим матрицу ошибок для текущего класса
		cm, _ := binary_metrics.GetConfusionMatrix(tmpTrue, tmpPred)
		tp, fn, fp := cm[1][1], cm[1][0], cm[0][1]

		// Затем в зависимости от параметра average надо это метрики усреднить
		switch average {
		case Macro:
			tmpRecall := float64(tp) / float64(tp+fn)
			tmpPrecision := float64(tp) / float64(tp+fp)
			tmpFScore := (1 + beta*beta) * tmpPrecision * tmpRecall / (beta*beta*tmpPrecision + tmpRecall)
			res += tmpFScore
		case Weighted:
			tmpRecall := float64(tp) / float64(tp+fn)
			tmpPrecision := float64(tp) / float64(tp+fp)
			tmpFScore := (1 + beta*beta) * tmpPrecision * tmpRecall / (beta*beta*tmpPrecision + tmpRecall)
			weightedFScore := float64(classCount[class]) * tmpFScore
			res += weightedFScore
		}
	}

	// В конце также надо усреднить ответ в зависимости от параметра
	switch average {
	case Macro:
		res /= float64(classesCount)
	case Weighted:
		res /= float64(len(yTrue))
	}

	if math.IsNaN(res) {
		return 0.0
	}

	return res
}
