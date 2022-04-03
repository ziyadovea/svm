package svc

import (
	"fmt"
	"testing"
)

func TestLinearKernel_Calculate(t *testing.T) {
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
			name: "Test",
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "241183.03",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := &LinearKernel{}
			if got := k.Calculate(tt.args.x, tt.args.y); fmt.Sprintf("%.2f", got) != tt.want {
				t.Errorf("Calculate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPolyKernel_Calculate(t *testing.T) {
	type fields struct {
		Coef0  float64
		Degree int64
	}
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   string
	}{
		{
			name: "Test1",
			fields: fields{
				Coef0:  0,
				Degree: 0,
			},
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "1.00",
		},
		{
			name: "Test2",
			fields: fields{
				Coef0:  0,
				Degree: 3,
			},
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "14029437694930058.00",
		},
		{
			name: "Test3",
			fields: fields{
				Coef0:  3,
				Degree: 0,
			},
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "1.00",
		},
		{
			name: "Test4",
			fields: fields{
				Coef0:  3,
				Degree: 2,
			},
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "58170703201.17",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := &PolyKernel{
				Coef0:  tt.fields.Coef0,
				Degree: tt.fields.Degree,
			}
			if got := k.Calculate(tt.args.x, tt.args.y); fmt.Sprintf("%.2f", got) != tt.want {
				t.Errorf("Calculate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRbfKernel_Calculate(t *testing.T) {
	type fields struct {
		Gamma float64
	}
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   string
	}{
		{
			name: "Test1",
			fields: fields{
				Gamma: 0,
			},
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "1.00",
		},
		{
			name: "Test2",
			fields: fields{
				Gamma: 1,
			},
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "0.00",
		},
		{
			name: "Test3",
			fields: fields{
				Gamma: 5,
			},
			args: args{
				x: []float64{1, 34, 324.23, 12.325, 543.21, -345.234, 23, 0},
				y: []float64{123.12, 213.12, 432.432, 54.234, 432.123, 432.324, 324.234, 23.23},
			},
			want: "0.00",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := &RbfKernel{
				Gamma: tt.fields.Gamma,
			}
			if got := k.Calculate(tt.args.x, tt.args.y); fmt.Sprintf("%.2f", got) != tt.want {
				t.Errorf("Calculate() = %v, want %v", got, tt.want)
			}
		})
	}
}
