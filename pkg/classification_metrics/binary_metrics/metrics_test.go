package binary_metrics

import (
	"fmt"
	"reflect"
	"testing"
)

func TestGetConfusionMatrix(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
	}
	tests := []struct {
		name  string
		args  args
		want  [2][2]int
		want1 string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
			},
			want: [2][2]int{
				{4, 4},
				{3, 7},
			},
			want1: `
Confusion matrix
 -------
| 4 | 4 |
 -------
| 3 | 7 |
 -------
`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := GetConfusionMatrix(tt.args.yTrue, tt.args.yPred)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetConfusionMatrix() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("GetConfusionMatrix() got = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestAccuracy(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
			},
			want: "0.611",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Accuracy(tt.args.yTrue, tt.args.yPred); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("Accuracy() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPrecision(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
			},
			want: "0.636",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Precision(tt.args.yTrue, tt.args.yPred); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("Precision() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRecall(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
			},
			want: "0.700",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Recall(tt.args.yTrue, tt.args.yPred); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("Recall() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFScore(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
			},
			want: "0.667",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FScore(tt.args.yTrue, tt.args.yPred); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("FScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFBetaScore(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
		beta  float64
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
				beta:  1,
			},
			want: "0.667",
		},
		{
			name: "Test2",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
				beta:  0.5,
			},
			want: "0.648",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FBetaScore(tt.args.yTrue, tt.args.yPred, tt.args.beta); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("ExtendedFScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBinaryClassificationReport(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
	}
	tests := []struct {
		name             string
		args             args
		wantCm           [2][2]int
		wantAccuracy     string
		wantPrecision    string
		wantRecall       string
		wantF1           string
		wantReportString string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
			},
			wantCm: [2][2]int{
				{4, 4},
				{3, 7},
			},
			wantAccuracy:  "0.611",
			wantPrecision: "0.636",
			wantRecall:    "0.700",
			wantF1:        "0.667",
			wantReportString: `
Confusion matrix
 -------
| 4 | 4 |
 -------
| 3 | 7 |
 -------

Accuracy  = 0.611
Precision = 0.636
Recall    = 0.700
F-score   = 0.667
`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCm, gotAccuracy, gotPrecision, gotRecall, gotF1, gotReportString := BinaryClassificationReport(tt.args.yTrue, tt.args.yPred)
			if !reflect.DeepEqual(gotCm, tt.wantCm) {
				t.Errorf("BinaryClassificationReport() gotCm = %v, want %v", gotCm, tt.wantCm)
			}
			if fmt.Sprintf("%.3f", gotAccuracy) != tt.wantAccuracy {
				t.Errorf("BinaryClassificationReport() gotAccuracy = %v, want %v", gotAccuracy, tt.wantAccuracy)
			}
			if fmt.Sprintf("%.3f", gotPrecision) != tt.wantPrecision {
				t.Errorf("BinaryClassificationReport() gotPrecision = %v, want %v", gotPrecision, tt.wantPrecision)
			}
			if fmt.Sprintf("%.3f", gotRecall) != tt.wantRecall {
				t.Errorf("BinaryClassificationReport() gotRecall = %v, want %v", gotRecall, tt.wantRecall)
			}
			if fmt.Sprintf("%.3f", gotF1) != tt.wantF1 {
				t.Errorf("BinaryClassificationReport() gotF1 = %v, want %v", gotF1, tt.wantF1)
			}
			if gotReportString != tt.wantReportString {
				t.Errorf("BinaryClassificationReport() gotReportString = %v, want %v", gotReportString, tt.wantReportString)
			}
		})
	}
}
