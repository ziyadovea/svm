package multiclass_metrics

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
	var tests = []struct {
		name  string
		args  args
		want  [][]int
		want1 string
	}{
		{
			name: "Test1",
			args: args{
				yTrue: []int{1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2},
				yPred: []int{1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 5, 1, 1},
			},
			want: [][]int{
				{3, 0, 0, 0, 1},
				{2, 1, 0, 1, 0},
				{0, 1, 3, 0, 0},
				{0, 1, 0, 0, 0},
				{0, 1, 1, 0, 0},
			},
			want1: `
Confusion matrix
3 0 0 0 1
2 1 0 1 0
0 1 3 0 0
0 1 0 0 0
0 1 1 0 0
`,
		},
		{
			name: "Test2",
			args: args{
				yTrue: []int{1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1},
				yPred: []int{1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1},
			},
			want: [][]int{
				{4, 4},
				{3, 7},
			},
			want1: `
Confusion matrix
4 4
3 7
`,
		},
		{
			name: "Test3",
			args: args{
				yTrue: []int{2, 0, 2, 2, 0, 1},
				yPred: []int{0, 0, 2, 2, 0, 2},
			},
			want: [][]int{
				{2, 0, 0},
				{0, 0, 1},
				{1, 0, 2},
			},
			want1: `
Confusion matrix
2 0 0
0 0 1
1 0 2
`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := GetConfusionMatrix(tt.args.yTrue, tt.args.yPred)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetConfusionMatrix() = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("GetConfusionMatrix() = %v, want %v", got1, tt.want1)
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
				yTrue: []int{0, 1, 2, 3, 4, 5},
				yPred: []int{0, 2, 2, 3, 4, 5},
			},
			want: "0.833",
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
		yTrue   []int
		yPred   []int
		average Average
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test macro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Macro,
			},
			want: "0.760",
		},
		{
			name: "Test micro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Micro,
			},
			want: "0.773",
		},
		{
			name: "Test weighted",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Weighted,
			},
			want: "0.812",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Precision(tt.args.yTrue, tt.args.yPred, tt.args.average); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("Precision() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRecall(t *testing.T) {
	type args struct {
		yTrue   []int
		yPred   []int
		average Average
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test macro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Macro,
			},
			want: "0.678",
		},
		{
			name: "Test micro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Micro,
			},
			want: "0.773",
		},
		{
			name: "Test weighted",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Weighted,
			},
			want: "0.773",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Recall(tt.args.yTrue, tt.args.yPred, tt.args.average); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("Recall() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFScore(t *testing.T) {
	type args struct {
		yTrue   []int
		yPred   []int
		average Average
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test macro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Macro,
			},
			want: "0.697",
		},
		{
			name: "Test micro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Micro,
			},
			want: "0.773",
		},
		{
			name: "Test weighted",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				average: Weighted,
			},
			want: "0.780",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FScore(tt.args.yTrue, tt.args.yPred, tt.args.average); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("FScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFBetaScore(t *testing.T) {
	type args struct {
		yTrue   []int
		yPred   []int
		beta    float64
		average Average
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1 macro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				beta:    1,
				average: Macro,
			},
			want: "0.697",
		},
		{
			name: "Test1 micro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				beta:    1,
				average: Micro,
			},
			want: "0.773",
		},
		{
			name: "Test1 weighted",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				beta:    1,
				average: Weighted,
			},
			want: "0.780",
		},
		{
			name: "Test2 macro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				beta:    0.5,
				average: Macro,
			},
			want: "0.727",
		},
		{
			name: "Test2 micro",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				beta:    0.5,
				average: Micro,
			},
			want: "0.773",
		},
		{
			name: "Test2 weighted",
			args: args{
				yTrue:   []int{2, 1, 3, 2, 4, 5, 3, 5, 2, 4, 2, 4, 3, 2, 4, 2, 4, 2, 2, 1, 4, 2},
				yPred:   []int{2, 2, 1, 2, 4, 1, 3, 5, 3, 4, 2, 4, 3, 2, 2, 2, 4, 2, 2, 1, 4, 2},
				beta:    0.5,
				average: Weighted,
			},
			want: "0.795",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FBetaScore(tt.args.yTrue, tt.args.yPred, tt.args.beta, tt.args.average); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("FBetaScore() = %v, want %v", got, tt.want)
			}
		})
	}
}
