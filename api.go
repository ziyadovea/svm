package svm

// Classifier - интерфейс для классификатора.
type Classifier interface {
	// Fit обучает модель на обучающей выборке.
	Fit(x [][]float64, y []int) error

	// Predict классифицирует входные данные на основе обученной модели.
	Predict(x [][]float64) []int

	// Clone возвращает копию текущего классификатора.
	Clone() (Classifier, error)
}
