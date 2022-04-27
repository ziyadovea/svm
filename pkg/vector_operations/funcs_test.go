package vector_operations

import (
	"fmt"
	"reflect"
	"testing"
)

func TestScalarProduct(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				x: []float64{},
				y: []float64{},
			},
			want: "0.00",
		},
		{
			name: "Test2",
			args: args{
				x: []float64{5, -4},
				y: []float64{2, 1},
			},
			want: "6.00",
		},
		{
			name: "Test3",
			args: args{
				x: []float64{0, 3},
				y: []float64{7, -1},
			},
			want: "-3.00",
		},
		{
			name: "Test4",
			args: args{
				x: []float64{5, 2},
				y: []float64{4, -1},
			},
			want: "18.00",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ScalarProduct(tt.args.x, tt.args.y); fmt.Sprintf("%.2f", got) != tt.want {
				t.Errorf("ScalarProduct() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				x: []float64{},
				y: []float64{},
			},
			want: "0.00",
		},
		{
			name: "Test2",
			args: args{
				x: []float64{1.4},
				y: []float64{5},
			},
			want: "3.60",
		},
		{
			name: "Test3",
			args: args{
				x: []float64{1.4, -324.5, 0, 34},
				y: []float64{45.7, 23, 2, -1.6},
			},
			want: "352.12",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EuclideanDistance(tt.args.x, tt.args.y); fmt.Sprintf("%.2f", got) != tt.want {
				t.Errorf("EuclideanDistance() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCountOfUniques(t *testing.T) {
	type args struct {
		x []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			name: "Test1",
			args: args{
				x: []int{},
			},
			want: 0,
		},
		{
			name: "Test2",
			args: args{
				x: []int{1},
			},
			want: 1,
		},
		{
			name: "Test3",
			args: args{
				x: []int{1, 1, 1},
			},
			want: 1,
		},
		{
			name: "Test4",
			args: args{
				x: []int{1, 2, 3},
			},
			want: 3,
		},
		{
			name: "Test5",
			args: args{
				x: []int{1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 3},
			},
			want: 3,
		},
		{
			name: "Test6",
			args: args{
				x: []int{1, 2, 1, 1, 1, 2, 2, 2, 1, 2},
			},
			want: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CountOfUniques(tt.args.x); got != tt.want {
				t.Errorf("CountOfUniques() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_isMatrixRectangular(t *testing.T) {
	type args struct {
		x [][]float64
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "Test1",
			args: args{
				x: [][]float64{
					{},
				},
			},
			want: true,
		},
		{
			name: "Test2",
			args: args{
				x: [][]float64{
					{1, 2, 3},
					{1, 2, 3},
					{1, 2, 3},
				},
			},
			want: true,
		},
		{
			name: "Test3",
			args: args{
				x: [][]float64{
					{1, 2, 3},
					{1, 2},
					{1, 2, 3},
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsMatrixRectangular(tt.args.x); got != tt.want {
				t.Errorf("isMatrixRectangular() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetUniques(t *testing.T) {
	type args struct {
		x []int
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{
			name: "Test1",
			args: args{
				x: []int{},
			},
			want: []int{},
		},
		{
			name: "Test2",
			args: args{
				x: []int{1, 2, 3},
			},
			want: []int{1, 2, 3},
		},
		{
			name: "Test3",
			args: args{
				x: []int{1, 1, 1, 2, 2},
			},
			want: []int{1, 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetUniques(tt.args.x); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetUniques() = %v, want %v", got, tt.want)
			}
		})
	}
}
