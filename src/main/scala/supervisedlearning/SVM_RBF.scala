package supervisedlearning

import weka.classifiers.Classifier
import weka.classifiers.functions.SMO
import weka.classifiers.functions.supportVector.RBFKernel

object SVM_RBF extends LearningAlgorithm {

  val name = "SVM: RBF Kernel"

  override def prep_classifier(x: Double): Classifier = {
    val smo = new SMO
    val rbf = new RBFKernel()
    rbf.setGamma(x)
    smo.setKernel(rbf)
    smo
  }

  override def x_values: List[Double] = {
    List(0.01,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00)
  }

  override def x_label: String = "gamma value"
}