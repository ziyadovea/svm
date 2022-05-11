package cross_validation

import (
	"fmt"
	"sort"
	"sync"

	"github.com/ziyadovea/svm"
	cls_metrics "github.com/ziyadovea/svm/pkg/classification_metrics"
	"github.com/ziyadovea/svm/pkg/classification_metrics/binary_metrics"
	"github.com/ziyadovea/svm/pkg/classification_metrics/multiclass_metrics"
	"github.com/ziyadovea/svm/pkg/vector_operations"
	"golang.org/x/sync/errgroup"
)

// KFoldCVScore реализует K-fold кросс валидацию.
// Возвращает мапу, где ключ - это метрика, значение - слайс значений этой метрики для каждого из разбиений.
// Не поддерживает метрику F beta score.
func KFoldCVScore(cls svm.Classifier, x [][]float64, y []int, nSplits int,
	metrics ...cls_metrics.ClassificationMetric) (map[cls_metrics.ClassificationMetric][]float64, error) {
	if nSplits < 2 {
		return nil, fmt.Errorf("nSpluts must be at least 2, actual: %d", nSplits)
	}

	// Проверим, является ли задача бинарной
	isBinary := vector_operations.IsBinary(y)
	// Далее профильтруем метрики в зависимости от типа задачи
	filteredMetrics := filterMetrics(isBinary, metrics...)

	res := make(map[cls_metrics.ClassificationMetric][]float64)
	cvData := KFoldCV(x, y, nSplits)

	eg := new(errgroup.Group)
	mu := sync.Mutex{}
	for _, data := range cvData {
		data := data
		cls, err := cls.Clone()
		if err != nil {
			return nil, err
		}

		eg.Go(func() error {
			err := cls.Fit(data.XTrain, data.YTrain)
			if err != nil {
				return err
			}

			yPred := cls.Predict(data.XTest)

			for _, metric := range filteredMetrics {
				mu.Lock()
				if _, ok := res[metric]; ok {
					res[metric] = append(res[metric], calculateMetric(data.YTest, yPred, metric))
				} else {
					res[metric] = []float64{calculateMetric(data.YTest, yPred, metric)}
				}
				mu.Unlock()
			}

			return nil
		})
	}

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	return res, nil
}

// CVData описывает набор данных, где xTrain и yTrain - данные для обучения, xTest и yTest - для валидации.
// Используется для k-fold кросс-валидации.
type CVData struct {
	XTrain [][]float64
	YTrain []int
	XTest  [][]float64
	YTest  []int
}

// KFoldCV возвращает слайс из наборов данных для обучения и валидации.
func KFoldCV(x [][]float64, y []int, nSplits int) []CVData {
	res := make([]CVData, nSplits)
	nSamples := len(y)

	// Надо узнать, сколько элементов у нас будет в тестовой выборке
	testSize := nSamples / nSplits
	beginTestSample := 0
	endTestSample := testSize - 1

	for i := 0; i < nSplits; i++ {
		cvData := CVData{
			XTrain: make([][]float64, 0, nSamples-testSize),
			YTrain: make([]int, 0, nSamples-testSize),
			XTest:  make([][]float64, 0, testSize),
			YTest:  make([]int, 0, testSize),
		}
		for j := 0; j < nSamples; j++ {
			if beginTestSample <= j && j <= endTestSample {
				cvData.XTest = append(cvData.XTest, x[j])
				cvData.YTest = append(cvData.YTest, y[j])
			} else {
				cvData.XTrain = append(cvData.XTrain, x[j])
				cvData.YTrain = append(cvData.YTrain, y[j])
			}

		}
		beginTestSample = endTestSample + 1
		endTestSample += testSize
		res[i] = cvData
	}

	return res
}

// Возвращает значение метрики для yTrue и yPred.
func calculateMetric(yTrue, yPred []int, metric cls_metrics.ClassificationMetric) float64 {
	res := 0.0
	switch metric {
	case cls_metrics.Accuracy:
		res = multiclass_metrics.Accuracy(yTrue, yPred)
	case cls_metrics.Precision:
		res = binary_metrics.Precision(yTrue, yPred)
	case cls_metrics.Recall:
		res = binary_metrics.Recall(yTrue, yPred)
	case cls_metrics.F1:
		res = binary_metrics.FScore(yTrue, yPred)
	case cls_metrics.PrecisionMacro:
		res = multiclass_metrics.Precision(yTrue, yPred, multiclass_metrics.Macro)
	case cls_metrics.RecallMacro:
		res = multiclass_metrics.Recall(yTrue, yPred, multiclass_metrics.Macro)
	case cls_metrics.F1Macro:
		res = multiclass_metrics.FScore(yTrue, yPred, multiclass_metrics.Macro)
	case cls_metrics.PrecisionMicro:
		res = multiclass_metrics.Precision(yTrue, yPred, multiclass_metrics.Micro)
	case cls_metrics.RecallMicro:
		res = multiclass_metrics.Recall(yTrue, yPred, multiclass_metrics.Micro)
	case cls_metrics.F1Micro:
		res = multiclass_metrics.FScore(yTrue, yPred, multiclass_metrics.Micro)
	case cls_metrics.PrecisionWeighted:
		res = multiclass_metrics.Precision(yTrue, yPred, multiclass_metrics.Weighted)
	case cls_metrics.RecallWeighted:
		res = multiclass_metrics.Recall(yTrue, yPred, multiclass_metrics.Weighted)
	case cls_metrics.F1Weighted:
		res = multiclass_metrics.FScore(yTrue, yPred, multiclass_metrics.Weighted)
	default: // По умолчанию, если метрика не определена, возвращаем метрику Accuracy
		res = multiclass_metrics.Accuracy(yTrue, yPred)
	}
	return res
}

// Фильтрует метрики в зависимости от типа задачи.
// Пример:
// Бинарная задачи:
// Если заданы метрики PrecisionMicro/Macro/Weighted, то в результат кладем только Precision.
// Мультиклассовая задача:
// Если задана метрика Precision, то вычисляются метрики PrecisionMicro/Macro/Weighted.
func filterMetrics(isBinary bool, metrics ...cls_metrics.ClassificationMetric) []cls_metrics.ClassificationMetric {
	set := make(map[cls_metrics.ClassificationMetric]struct{}, 0)
	for _, metric := range metrics {
		if isBinary {
			switch metric {
			case cls_metrics.Accuracy:
				set[cls_metrics.Accuracy] = struct{}{}
			case cls_metrics.Precision:
				set[cls_metrics.Precision] = struct{}{}
			case cls_metrics.Recall:
				set[cls_metrics.Recall] = struct{}{}
			case cls_metrics.F1:
				set[cls_metrics.F1] = struct{}{}
			case cls_metrics.PrecisionMacro:
				fallthrough
			case cls_metrics.PrecisionMicro:
				fallthrough
			case cls_metrics.PrecisionWeighted:
				set[cls_metrics.Precision] = struct{}{}
			case cls_metrics.RecallMacro:
				fallthrough
			case cls_metrics.RecallMicro:
				fallthrough
			case cls_metrics.RecallWeighted:
				set[cls_metrics.Recall] = struct{}{}
			case cls_metrics.F1Macro:
				fallthrough
			case cls_metrics.F1Micro:
				fallthrough
			case cls_metrics.F1Weighted:
				set[cls_metrics.F1] = struct{}{}
			}
		} else {
			switch metric {
			case cls_metrics.Accuracy:
				set[cls_metrics.Accuracy] = struct{}{}
			case cls_metrics.Precision:
				set[cls_metrics.PrecisionMicro] = struct{}{}
				set[cls_metrics.PrecisionMacro] = struct{}{}
				set[cls_metrics.PrecisionWeighted] = struct{}{}
			case cls_metrics.Recall:
				set[cls_metrics.RecallMicro] = struct{}{}
				set[cls_metrics.RecallMacro] = struct{}{}
				set[cls_metrics.RecallWeighted] = struct{}{}
			case cls_metrics.F1:
				set[cls_metrics.F1Micro] = struct{}{}
				set[cls_metrics.F1Macro] = struct{}{}
				set[cls_metrics.F1Weighted] = struct{}{}
			case cls_metrics.PrecisionMacro:
				set[cls_metrics.PrecisionMacro] = struct{}{}
			case cls_metrics.RecallMacro:
				set[cls_metrics.RecallMacro] = struct{}{}
			case cls_metrics.F1Macro:
				set[cls_metrics.F1Macro] = struct{}{}
			case cls_metrics.PrecisionMicro:
				set[cls_metrics.PrecisionMicro] = struct{}{}
			case cls_metrics.RecallMicro:
				set[cls_metrics.RecallMicro] = struct{}{}
			case cls_metrics.F1Micro:
				set[cls_metrics.F1Micro] = struct{}{}
			case cls_metrics.PrecisionWeighted:
				set[cls_metrics.PrecisionWeighted] = struct{}{}
			case cls_metrics.RecallWeighted:
				set[cls_metrics.RecallWeighted] = struct{}{}
			case cls_metrics.F1Weighted:
				set[cls_metrics.F1Weighted] = struct{}{}
			}
		}
	}

	res := make([]cls_metrics.ClassificationMetric, 0, len(set))
	for k, _ := range set {
		res = append(res, k)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i] < res[j]
	})

	return res
}
