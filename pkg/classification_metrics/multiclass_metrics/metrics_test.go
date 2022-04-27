package multiclass_metrics

import (
	"reflect"
	"testing"
)

func TestGetConfusionMatrix(t *testing.T) {
	type args struct {
		yTrue []int
		yPred []int
	}
	var tests = []struct {
		name string
		args args
		want [][]int
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
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetConfusionMatrix(tt.args.yTrue, tt.args.yPred); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetConfusionMatrix() = %v, want %v", got, tt.want)
			}
		})
	}
}
