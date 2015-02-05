package supervisedlearning

import weka.classifiers.Classifier
import weka.classifiers.functions.MultilayerPerceptron

object NeuralNetwork extends LearningAlgorithm {

  val name = "Neural Network: MultilayerPerceptron"

  override def prep_classifier(x: Double): Classifier = {
    val mp = new MultilayerPerceptron
    mp.setTrainingTime(x.toInt)
    mp
  }

  override def x_values: List[Double] = (1 to 10).toList.map(_.toDouble)

  override def x_label: String = "# Epoches"
}