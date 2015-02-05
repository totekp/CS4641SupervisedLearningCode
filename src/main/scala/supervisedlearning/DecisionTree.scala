package supervisedlearning

import weka.classifiers.Classifier
import weka.classifiers.trees.J48




object DecisionTree extends LearningAlgorithm {

  val name = "Decision Tree: J48/C4.5"

  override def prep_classifier(x: Double): Classifier = {
    val j48 = new J48
    j48
  }

  override def x_values: List[Double] = (10 to 100 by 10).toList.map(_.toDouble)

  override def x_label: String = "training data %"

  override def x_to_train_percentage(x: Double): Double = x.toDouble / 100
}

