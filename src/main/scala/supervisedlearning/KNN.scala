package supervisedlearning

import weka.classifiers.Classifier
import weka.classifiers.`lazy`.IBk
import weka.core.neighboursearch.KDTree

object KNN extends LearningAlgorithm {

  val name = "kNN"

  override def prep_classifier(x: Double): Classifier = {
    val knn = new IBk()
//    knn.setNearestNeighbourSearchAlgorithm(new KDTree())
    knn.setKNN(x.toInt)
    knn
  }

  override def x_values: List[Double] = (1 to 10).toList.map(_.toDouble)

  override def x_label: String = "Value of k"
}