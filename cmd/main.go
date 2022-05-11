package main

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/ziyadovea/svm/pkg/classification_metrics"
	"github.com/ziyadovea/svm/pkg/cross_validation"
	"github.com/ziyadovea/svm/svc"
)

func main() {
	// simple test for binary
	X := [][]float64{
		[]float64{1.3, 2.1}, []float64{-1.5, -1.}, []float64{-1.4, -0.9}, []float64{-1.3, -1.2}, []float64{-1.1, -0.2}, []float64{-1.2, -0.4}, []float64{-0.5, 1.2},
		[]float64{-1.5, 2.1}, []float64{1., 1.}, []float64{1.3, 0.8}, []float64{1.2, 0.5}, []float64{0.2, -2.}, []float64{0.5, -2.4}, []float64{0.2, -2.3}, []float64{0., -2.7},
		[]float64{1.3, 2.1},
	}
	Y := []int{1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1}

	cls := svc.NewSVC()
	res, _ := cross_validation.KFoldCVScore(cls, X, Y, 5, classification_metrics.Accuracy)
	fmt.Println(res)
	//err := cls.Fit(X, Y)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//yPred := cls.Predict(X)
	//fmt.Printf("y: %v\n", Y)
	//fmt.Printf("y pred: %v\n", yPred)
	//fmt.Printf("acc: %f\n", binary_metrics.Accuracy(Y, yPred))

	// simple test for multiclass
	//records, err := readData("./datasets/iris_headers.csv")
	//if err != nil {
	//	log.Fatal(err)
	//}
	//x := make([][]float64, len(records))
	//for i := 0; i < len(x); i++ {
	//	x[i] = make([]float64, 4)
	//}
	//y := make([]int, len(records))
	//for i, record := range records {
	//	x[i][0], _ = strconv.ParseFloat(record[0], 64)
	//	x[i][1], _ = strconv.ParseFloat(record[1], 64)
	//	x[i][2], _ = strconv.ParseFloat(record[2], 64)
	//	x[i][3], _ = strconv.ParseFloat(record[3], 64)
	//
	//	switch record[4] {
	//	case "Iris-virginica":
	//		y[i] = 2
	//	case "Iris-versicolor":
	//		y[i] = 1
	//	case "Iris-setosa":
	//		y[i] = 0
	//	}
	//}
	//cls := svc.NewMultiSVC()
	//err = cls.Fit(x, y)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//yPred := cls.Predict(x)
	//fmt.Printf("y: %v\n", y)
	//fmt.Printf("y pred: %v\n", yPred)
	//fmt.Printf("acc: %f\n", multiclass_metrics.Accuracy(y, yPred))
}

func readData(fileName string) ([][]string, error) {

	f, err := os.Open(fileName)

	if err != nil {
		return [][]string{}, err
	}

	defer f.Close()

	r := csv.NewReader(f)

	// skip first line
	if _, err := r.Read(); err != nil {
		return [][]string{}, err
	}

	records, err := r.ReadAll()

	if err != nil {
		return [][]string{}, err
	}

	return records, nil
}
