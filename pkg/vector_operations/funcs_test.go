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

func TestCount(t *testing.T) {
	type args struct {
		x  []int
		el int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			name: "Test1",
			args: args{
				x:  []int{1, 2, 3},
				el: 0,
			},
			want: 0,
		},
		{
			name: "Test2",
			args: args{
				x:  []int{1, 2, 3},
				el: 1,
			},
			want: 1,
		},
		{
			name: "Test3",
			args: args{
				x:  []int{1, 1, 1},
				el: 1,
			},
			want: 3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Count(tt.args.x, tt.args.el); got != tt.want {
				t.Errorf("Count() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCounter(t *testing.T) {
	type args struct {
		x []int
	}
	tests := []struct {
		name string
		args args
		want map[int]int
	}{
		{
			name: "Test1",
			args: args{
				x: []int{1, 2, 3, 1, 1, 2, 2, 2},
			},
			want: map[int]int{
				1: 3,
				2: 4,
				3: 1,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Counter(tt.args.x); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Counter() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAverage(t *testing.T) {
	type args struct {
		x []float64
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test1",
			args: args{
				x: []float64{2, 4, 6, 8, 10},
			},
			want: "6.000",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Average(tt.args.x); fmt.Sprintf("%.3f", got) != tt.want {
				t.Errorf("Average() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsBinary(t *testing.T) {
	type args struct {
		x []int
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "Test ok",
			args: args{
				x: []int{0, 1, 1, 1, 0, 1, 1},
			},
			want: true,
		},
		{
			name: "Test ok only 1",
			args: args{
				x: []int{1, 1, 1, 1, 1, 1, 1},
			},
			want: true,
		},
		{
			name: "Test ok only 0",
			args: args{
				x: []int{0, 0, 0, 0},
			},
			want: true,
		},
		{
			name: "Test not ok",
			args: args{
				x: []int{0, 2, 1, 1, 1, 0, 1, 1},
			},
			want: true, // todo
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsBinary(tt.args.x); got != tt.want {
				t.Errorf("IsBinary() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSortByValue(t *testing.T) {
	type args struct {
		m map[int]float64
	}
	tests := []struct {
		name string
		args args
		want PairList
	}{
		{
			name: "Test1",
			args: args{
				m: map[int]float64{
					1: 34,
					2: 12,
					3: 56,
					4: 2,
					5: 35,
				},
			},
			want: []Pair{
				{3, 56},
				{5, 35},
				{1, 34},
				{2, 12},
				{4, 2},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SortByValue(tt.args.m); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("SortByValue() = %v, want %v", got, tt.want)
			}
		})
	}
}
