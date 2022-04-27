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
		name string
		args args
		want [2][2]int
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
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := GetConfusionMatrix(tt.args.yTrue, tt.args.yPred)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetConfusionMatrix() got = %v, want %v", got, tt.want)
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

func TestExtendedFScore(t *testing.T) {
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ExtendedFScore(tt.args.yTrue, tt.args.yPred, tt.args.beta); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("ExtendedFScore() = %v, want %v", got, tt.want)
			}
		})
	}
}
