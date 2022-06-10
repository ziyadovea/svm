package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/xuri/excelize/v2"
	"github.com/ziyadovea/svm"
	"github.com/ziyadovea/svm/pkg/classification_metrics"
	"github.com/ziyadovea/svm/pkg/classification_metrics/multiclass_metrics"
	"github.com/ziyadovea/svm/pkg/cross_validation"
	"github.com/ziyadovea/svm/pkg/vector_operations"
	"github.com/ziyadovea/svm/svc"
)

func main() {
	currDur, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	pathToTrainDataTmpl := filepath.Join(currDur, "datasets", "%d_train_data.xlsx")
	pathToTestDataTmpl := filepath.Join(currDur, "datasets", "%d_test_data.xlsx")

	reportFile, err := os.Create("report.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := reportFile.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	for _, wellID := range []int{25, 336, 2697} {
		reportFile.WriteString(fmt.Sprintf("REPORT FOR WELL %d\n\n", wellID))

		pathToTrainData := fmt.Sprintf(pathToTrainDataTmpl, wellID)
		pathToTestData := fmt.Sprintf(pathToTestDataTmpl, wellID)

		xTrain, yTrain, err := readData(pathToTrainData)
		if err != nil {
			log.Fatal(err)
		}

		xTest, yTest, err := readData(pathToTestData)
		if err != nil {
			log.Fatal(err)
		}

		// test 1
		cls := svc.NewMultiSVC()
		if err = cls.SetKernelByName("linear"); err != nil {
			log.Fatal(err)
		}
		if err = testCls(cls, xTrain, xTest, yTrain, yTest, reportFile); err != nil {
			log.Fatal(err)
		}

		// test 2
		cls = svc.NewMultiSVC()
		if err = cls.SetKernelByName("poly"); err != nil {
			log.Fatal(err)
		}
		if err = testCls(cls, xTrain, xTest, yTrain, yTest, reportFile); err != nil {
			log.Fatal(err)
		}

		// test 3
		cls = svc.NewMultiSVC()
		if err = cls.SetKernelByName("poly"); err != nil {
			log.Fatal(err)
		}
		cls.C = 0.1
		cls.Degree = 5
		if err = testCls(cls, xTrain, xTest, yTrain, yTest, reportFile); err != nil {
			log.Fatal(err)
		}

		// test 4
		cls = svc.NewMultiSVC()
		if err = cls.SetKernelByName("poly"); err != nil {
			log.Fatal(err)
		}
		cls.C = 10
		cls.Degree = 5
		if err = testCls(cls, xTrain, xTest, yTrain, yTest, reportFile); err != nil {
			log.Fatal(err)
		}

		// test 5
		cls = svc.NewMultiSVC()
		if err = cls.SetKernelByName("rbf"); err != nil {
			log.Fatal(err)
		}
		if err = testCls(cls, xTrain, xTest, yTrain, yTest, reportFile); err != nil {
			log.Fatal(err)
		}

		// test 6
		cls = svc.NewMultiSVC()
		if err = cls.SetKernelByName("rbf"); err != nil {
			log.Fatal(err)
		}
		cls.C = 0.1
		cls.Gamma = 0.1
		if err = testCls(cls, xTrain, xTest, yTrain, yTest, reportFile); err != nil {
			log.Fatal(err)
		}

		// test 7
		cls = svc.NewMultiSVC()
		if err = cls.SetKernelByName("rbf"); err != nil {
			log.Fatal(err)
		}
		cls.C = 10
		cls.Gamma = 10
		if err = testCls(cls, xTrain, xTest, yTrain, yTest, reportFile); err != nil {
			log.Fatal(err)
		}
	}
}

func readData(fileName string) ([][]float64, []int, error) {
	f, err := excelize.OpenFile(fileName)
	if err != nil {
		return nil, nil, err
	}
	defer func() {
		if err := f.Close(); err != nil {
			log.Println(err)
		}
	}()

	cols, err := f.GetCols("Sheet1")
	if err != nil {
		return nil, nil, err
	}

	x := make([][]float64, len(cols[0])-1)
	for i := range x {
		x[i] = make([]float64, len(cols)-1)
	}
	y := make([]int, len(cols[0])-1)

	for i := 0; i < len(cols); i++ {
		for j := 1; j < len(cols[i]); j++ {
			switch cols[i][0] {
			case "Liquid":
				x[j-1][0], err = strconv.ParseFloat(cols[i][j], 64)
				if err != nil {
					return nil, nil, err
				}
			case "Gas":
				x[j-1][1], err = strconv.ParseFloat(cols[i][j], 64)
				if err != nil {
					return nil, nil, err
				}
			case "Water cut":
				x[j-1][2], err = strconv.ParseFloat(cols[i][j], 64)
				if err != nil {
					return nil, nil, err
				}
			case "Label":
				y[j-1], err = strconv.Atoi(cols[i][j])
				if err != nil {
					return nil, nil, err
				}
			}
		}
	}

	return x, y, nil
}

func testCls(cls svm.Classifier, xTrain, xTest [][]float64, yTrain, yTest []int, sw io.StringWriter) error {
	sw.WriteString("NEW CLASSIFIER REPORT\n\n")

	// CV
	now := time.Now()
	scores, err := cross_validation.KFoldCVScore(cls, xTrain, yTrain, 5, classification_metrics.Accuracy, classification_metrics.F1)
	if err != nil {
		return err
	}
	sw.WriteString(fmt.Sprintf("CV duration: %s\n\n", time.Since(now).String()))
	for k, v := range scores {
		sw.WriteString(fmt.Sprintf("Metric %s:\n", string(k)))
		sw.WriteString(fmt.Sprintf("Scores %+v:\n", v))
		sw.WriteString(fmt.Sprintf("Avg scores: %f\n", vector_operations.Average(v)))
		sw.WriteString("\n")
	}
	sw.WriteString("---\n")

	// FIT
	now = time.Now()
	if err = cls.Fit(xTrain, yTrain); err != nil {
		return err
	}
	sw.WriteString(fmt.Sprintf("Fit duration: %s\n", time.Since(now).String()))
	sw.WriteString("---\n")

	// PREDICT
	yPred := cls.Predict(xTest)
	sw.WriteString(fmt.Sprintf("Accuracy: %f\n", multiclass_metrics.Accuracy(yTest, yPred)))
	sw.WriteString(fmt.Sprintf("F1 micro: %f\n", multiclass_metrics.FScore(yTest, yPred, multiclass_metrics.Micro)))
	sw.WriteString(fmt.Sprintf("F1 macro: %f\n", multiclass_metrics.FScore(yTest, yPred, multiclass_metrics.Macro)))
	sw.WriteString(fmt.Sprintf("F1 weighted: %f\n", multiclass_metrics.FScore(yTest, yPred, multiclass_metrics.Weighted)))
	sw.WriteString("---\n")

	return nil
}
