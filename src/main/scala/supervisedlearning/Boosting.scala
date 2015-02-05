package supervisedlearning

import weka.classifiers.Classifier
import weka.classifiers.meta.AdaBoostM1
import weka.classifiers.trees.J48

object Boosting extends LearningAlgorithm {

  val name = "Boosting: AdaBoostM1 on J48"

  override def prep_classifier(x: Double): Classifier = {
    val ada = new AdaBoostM1
    ada.setClassifier(new J48())
    ada.setNumIterations(x.toInt)
    ada
  }

  override def x_values: List[Double] = (1 to 10).toList.map(_.toDouble)

  override def x_label: String = "# Iterations"
}