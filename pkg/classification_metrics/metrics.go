package classification_metrics

type ClassificationMetric string

const (
	// binary metrics
	Accuracy  ClassificationMetric = "accuracy"
	Precision ClassificationMetric = "precision"
	Recall    ClassificationMetric = "recall"
	F1        ClassificationMetric = "f1"
	FBeta     ClassificationMetric = "f1_beta"

	// multiclass metrics
	// macro average
	PrecisionMacro ClassificationMetric = "precision_macro"
	RecallMacro    ClassificationMetric = "recall_macro"
	F1Macro        ClassificationMetric = "f1_macro"
	FBetaMacro     ClassificationMetric = "f1_beta_macro"

	// micro average
	PrecisionMicro ClassificationMetric = "precision_micro"
	RecallMicro    ClassificationMetric = "recall_micro"
	F1Micro        ClassificationMetric = "f1_micro"
	FBetaMicro     ClassificationMetric = "f1_beta_micro"

	// weighted average
	PrecisionWeighted ClassificationMetric = "precision_weighted"
	RecallWeighted    ClassificationMetric = "recall_weighted"
	F1Weighted        ClassificationMetric = "f1_weighted"
	FBetaWeighted     ClassificationMetric = "f1_beta_weighted"
)
