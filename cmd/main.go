package main

import (
	"fmt"
	"github.com/ziyadovea/svm/pkg/classification_metrics/multiclass_metrics"
)

func main() {
	//X := [][]float64{
	//	[]float64{0.4, -0.7}, []float64{-1.5, -1.}, []float64{-1.4, -0.9}, []float64{-1.3, -1.2}, []float64{-1.1, -0.2}, []float64{-1.2, -0.4}, []float64{-0.5, 1.2},
	//	[]float64{-1.5, 2.1}, []float64{1., 1.}, []float64{1.3, 0.8}, []float64{1.2, 0.5}, []float64{0.2, -2.}, []float64{0.5, -2.4}, []float64{0.2, -2.3}, []float64{0., -2.7},
	//	[]float64{1.3, 2.1},
	//}
	//Y := []int{-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1}
	//
	//cls := svc.NewSVC()
	//err := cls.Fit(X, Y)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//yPred := cls.Predict(X)
	//fmt.Printf("y: %v\n", Y)
	//fmt.Printf("y pred: %v\n", yPred)
	//fmt.Printf("acc: %f\n", binary_metrics.Accuracy(Y, yPred))
	//_, cm := binary_metrics.GetConfusionMatrix(Y, yPred)
	//fmt.Println(cm)
	fmt.Printf("%v",
		multiclass_metrics.GetConfusionMatrix(
			[]int{1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2},
			[]int{1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 5, 1, 1}),
	)
}
