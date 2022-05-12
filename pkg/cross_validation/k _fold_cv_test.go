package cross_validation

import (
	"errors"
	"fmt"
	"github.com/ziyadovea/svm"
	cls_metrics "github.com/ziyadovea/svm/pkg/classification_metrics"
	"reflect"
	"sort"
	"testing"
)

func Test_filterMetrics(t *testing.T) {
	type args struct {
		isBinary bool
		metrics  []cls_metrics.ClassificationMetric
	}
	tests := []struct {
		name string
		args args
		want []cls_metrics.ClassificationMetric
	}{
		{
			name: "Test is binary only binary metrics",
			args: args{
				isBinary: true,
				metrics: []cls_metrics.ClassificationMetric{
					cls_metrics.Accuracy,
					cls_metrics.Recall,
					cls_metrics.Precision,
					cls_metrics.F1,
				},
			},
			want: []cls_metrics.ClassificationMetric{
				cls_metrics.Accuracy,
				cls_metrics.F1,
				cls_metrics.Precision,
				cls_metrics.Recall,
			},
		},
		{
			name: "Test is binary not only binary metrics",
			args: args{
				isBinary: true,
				metrics: []cls_metrics.ClassificationMetric{
					cls_metrics.RecallMicro,
					cls_metrics.RecallMacro,
					cls_metrics.RecallWeighted,
					cls_metrics.Recall,
				},
			},
			want: []cls_metrics.ClassificationMetric{
				cls_metrics.Recall,
			},
		},
		{
			name: "Test is multiclass only multiclass metrics",
			args: args{
				isBinary: false,
				metrics: []cls_metrics.ClassificationMetric{
					cls_metrics.RecallMicro,
					cls_metrics.RecallMacro,
					cls_metrics.RecallWeighted,
				},
			},
			want: []cls_metrics.ClassificationMetric{
				cls_metrics.RecallMacro,
				cls_metrics.RecallMicro,
				cls_metrics.RecallWeighted,
			},
		},
		{
			name: "Test is multiclass not only multiclass metrics",
			args: args{
				isBinary: false,
				metrics: []cls_metrics.ClassificationMetric{
					cls_metrics.RecallMicro,
					cls_metrics.RecallMacro,
					cls_metrics.RecallWeighted,
					cls_metrics.Recall,
				},
			},
			want: []cls_metrics.ClassificationMetric{
				cls_metrics.RecallMacro,
				cls_metrics.RecallMicro,
				cls_metrics.RecallWeighted,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := filterMetrics(tt.args.isBinary, tt.args.metrics...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("filterMetrics() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestKFoldCV(t *testing.T) {
	type args struct {
		x       [][]float64
		y       []int
		nSplits int
	}
	tests := []struct {
		name string
		args args
		want []CVData
	}{
		{
			name: "Test1",
			args: args{
				x: [][]float64{
					{1, 1, 1},
					{2, 2, 2},
					{3, 3, 3},
					{4, 4, 4},
					{5, 5, 5},
					{6, 6, 6},
					{7, 7, 7},
					{8, 8, 8},
					{9, 9, 9},
					{10, 10, 10},
					{11, 11, 11},
					{12, 12, 12},
				},
				y:       []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				nSplits: 3,
			},
			want: []CVData{
				{
					XTrain: [][]float64{
						{5, 5, 5},
						{6, 6, 6},
						{7, 7, 7},
						{8, 8, 8},
						{9, 9, 9},
						{10, 10, 10},
						{11, 11, 11},
						{12, 12, 12},
					},
					YTrain: []int{5, 6, 7, 8, 9, 10, 11, 12},
					XTest: [][]float64{
						{1, 1, 1},
						{2, 2, 2},
						{3, 3, 3},
						{4, 4, 4},
					},
					YTest: []int{1, 2, 3, 4},
				},
				{
					XTrain: [][]float64{
						{1, 1, 1},
						{2, 2, 2},
						{3, 3, 3},
						{4, 4, 4},
						{9, 9, 9},
						{10, 10, 10},
						{11, 11, 11},
						{12, 12, 12},
					},
					YTrain: []int{1, 2, 3, 4, 9, 10, 11, 12},
					XTest: [][]float64{
						{5, 5, 5},
						{6, 6, 6},
						{7, 7, 7},
						{8, 8, 8},
					},
					YTest: []int{5, 6, 7, 8},
				},
				{
					XTrain: [][]float64{
						{1, 1, 1},
						{2, 2, 2},
						{3, 3, 3},
						{4, 4, 4},
						{5, 5, 5},
						{6, 6, 6},
						{7, 7, 7},
						{8, 8, 8},
					},
					YTrain: []int{1, 2, 3, 4, 5, 6, 7, 8},
					XTest: [][]float64{
						{9, 9, 9},
						{10, 10, 10},
						{11, 11, 11},
						{12, 12, 12},
					},
					YTest: []int{9, 10, 11, 12},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := KFoldCV(tt.args.x, tt.args.y, tt.args.nSplits); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("KFoldCV() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestKFoldCVScore(t *testing.T) {
	type args struct {
		cls     svm.Classifier
		x       [][]float64
		y       []int
		nSplits int
		metrics []cls_metrics.ClassificationMetric
	}
	tests := []struct {
		name    string
		args    args
		want    map[cls_metrics.ClassificationMetric][]float64
		wantErr bool
	}{
		{
			name: "Test nSplits < 2",
			args: args{
				cls:     nil,
				x:       nil,
				y:       nil,
				nSplits: 0,
				metrics: nil,
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "Test classifier error",
			args: args{
				cls: &MockClassifier{fitImpl: func(x [][]float64, y []int) error {
					return errors.New("")
				}},
				x: [][]float64{
					{1, 1, 1},
					{2, 2, 2},
					{3, 3, 3},
					{4, 4, 4},
					{5, 5, 5},
					{6, 6, 6},
					{7, 7, 7},
					{8, 8, 8},
					{9, 9, 9},
					{10, 10, 10},
					{11, 11, 11},
					{12, 12, 12},
				},
				y:       []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				nSplits: 3,
				metrics: nil,
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "Test classifier accuracy",
			args: args{
				cls: &MockClassifier{
					fitImpl: func(x [][]float64, y []int) error {
						return nil
					},
					predictImpl: func(x [][]float64) []int {
						return []int{1, 2, 3, 4, 9, 10, 11, 12}
					},
				},
				x: [][]float64{
					{1, 1, 1},
					{2, 2, 2},
					{3, 3, 3},
					{4, 4, 4},
					{5, 5, 5},
					{6, 6, 6},
					{7, 7, 7},
					{8, 8, 8},
					{9, 9, 9},
					{10, 10, 10},
					{11, 11, 11},
					{12, 12, 12},
				},
				y:       []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				nSplits: 3,
				metrics: []cls_metrics.ClassificationMetric{cls_metrics.Accuracy},
			},
			want: map[cls_metrics.ClassificationMetric][]float64{
				cls_metrics.Accuracy: []float64{0, 1, 0},
			},
			wantErr: false,
		},
		{
			name: "Test classifier recall, precision-micro, accuracy",
			args: args{
				cls: &MockClassifier{
					fitImpl: func(x [][]float64, y []int) error {
						return nil
					},
					predictImpl: func(x [][]float64) []int {
						return []int{1, 2, 9, 12}
					},
				},
				x: [][]float64{
					{1, 1, 1},
					{2, 2, 2},
					{3, 3, 3},
					{4, 4, 4},
					{5, 5, 5},
					{6, 6, 6},
					{7, 7, 7},
					{8, 8, 8},
					{9, 9, 9},
					{10, 10, 10},
					{11, 11, 11},
					{12, 12, 12},
				},
				y:       []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				nSplits: 3,
				metrics: []cls_metrics.ClassificationMetric{cls_metrics.Recall, cls_metrics.Precision, cls_metrics.F1, cls_metrics.Accuracy},
			},
			want: map[cls_metrics.ClassificationMetric][]float64{
				cls_metrics.Accuracy: []float64{0.25, 0.5, 0},

				cls_metrics.RecallMicro:    []float64{0.25, 0.5, 0},
				cls_metrics.RecallMacro:    []float64{0.25, 0.5, 0},
				cls_metrics.RecallWeighted: []float64{0.25, 0.5, 0},

				cls_metrics.PrecisionMicro:    []float64{0, 0.5, 1},
				cls_metrics.PrecisionMacro:    []float64{0, 0, 0},
				cls_metrics.PrecisionWeighted: []float64{0, 0, 0},

				cls_metrics.F1Micro:    []float64{0.667, 0, 0.333},
				cls_metrics.F1Macro:    []float64{0, 0, 0},
				cls_metrics.F1Weighted: []float64{0, 0, 0},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := KFoldCVScore(tt.args.cls, tt.args.x, tt.args.y, tt.args.nSplits, tt.args.metrics...)
			if (err != nil) != tt.wantErr {
				t.Errorf("KFoldCVScore() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmpMapFloats(got, tt.want) {
				t.Errorf("KFoldCVScore() got = %v, want %v", got, tt.want)
			}
		})
	}
}

var _ svm.Classifier = (*MockClassifier)(nil)

type MockClassifier struct {
	fitImpl     func(x [][]float64, y []int) error
	predictImpl func(x [][]float64) []int
}

func (m *MockClassifier) Clone() (svm.Classifier, error) {
	return m, nil
}

func (m *MockClassifier) Fit(x [][]float64, y []int) error {
	return m.fitImpl(x, y)
}

func (m *MockClassifier) Predict(x [][]float64) []int {
	return m.predictImpl(x)
}

// Сравнивает две мапы, где ключ - тип метрики классификации, значение -  слайс float64.
// Сравнение происходит независимо от порядка ключей в мапе и независимо от порядка элементов в слайсе.
func cmpMapFloats(got, want map[cls_metrics.ClassificationMetric][]float64) bool {
	for k, v := range want {
		if val, ok := got[k]; ok {

			sort.Slice(v, func(i, j int) bool {
				return v[i] < v[j]
			})

			sort.Slice(val, func(i, j int) bool {
				return val[i] < val[j]
			})

			for i, item := range v {
				if fmt.Sprintf("%.3f", item) != fmt.Sprintf("%.3f", val[i]) {
					return false
				}
			}
		} else {
			return false
		}
	}
	return true
}
