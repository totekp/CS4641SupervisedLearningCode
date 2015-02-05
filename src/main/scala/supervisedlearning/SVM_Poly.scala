package supervisedlearning

import weka.classifiers.Classifier
import weka.classifiers.functions.SMO
import weka.classifiers.functions.supportVector.PolyKernel

object SVM_Poly extends LearningAlgorithm {

  val name = "SVM: Poly Kernel"

  override def prep_classifier(x: Double): Classifier = {
    val smo = new SMO
    val poly = new PolyKernel()
    poly.setExponent(x)
    smo.setKernel(poly)
    smo
  }

  override def x_values: List[Double] = (1 to 10).toList.map(_.toDouble)

  override def x_label: String = "Exponent"
}