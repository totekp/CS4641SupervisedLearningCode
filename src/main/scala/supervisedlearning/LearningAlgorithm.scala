package supervisedlearning

import java.io.PrintWriter
import java.security.SecureRandom

import weka.classifiers.{Classifier, Evaluation}
import weka.core.Instances

import scala.math.BigDecimal.RoundingMode

trait LearningAlgorithm {
  def name: String
  def run(pw: PrintWriter, _dataset: Instances): Unit = {

    pw.println("Learning algorithm: %s".format(name))
    val dataset = new Instances(_dataset)
    dataset.randomize(new SecureRandom())

    val result = Array.fill(4, x_values.size + 1)("")
    result(0)(0) = x_label
    result(1)(0) = "training error %"
    result(2)(0) = "testing error %"
    result(3)(0) = "training time (ms)"

    x_values.zipWithIndex.foreach {
      case (x, _i) =>

        val i = _i + 1
        println("%s: %s".format(name, x))
        result(0)(i) = x.toString
        val (train, test) = getTrainingAndTest(x_to_train_percentage(x), dataset)
        val classifier = prep_classifier(x)
        result(3)(i) = timer_seconds(classifier.buildClassifier(train)).toString
        result(1)(i) = errorPercentage(classifier, train, train)
        result(2)(i) = errorPercentage(classifier, train, test)
    }
    Runner.outputStringMatrix(result)
  }

  def timer_seconds(block: => Unit): Double = {
    (0 until 5).foreach(i => System.currentTimeMillis()) // warm up
    val a = System.currentTimeMillis()
    block
    val diff = System.currentTimeMillis() - a
    BigDecimal(diff).setScale(2, RoundingMode.HALF_EVEN).doubleValue()
  }

  def prep_classifier(x: Double): Classifier

  def x_label: String
  def x_values: List[Double]

  // take 66% as training
  def x_to_train_percentage(x: Double): Double = 0.66

  def getTrainingAndTest(train_percent: Double, dataset: Instances): (Instances, Instances) = {
    val total = dataset.size()
    val trainSize = (total * train_percent).toInt
    val testSize = total - trainSize
    val train = new Instances(dataset, 0, trainSize)
    val test = new Instances(dataset, trainSize, testSize)
    (train, test)
  }

  def errorPercentage(cf: Classifier, train: Instances, test: Instances): String = {
    val eval: Evaluation = new Evaluation(train)
    eval.evaluateModel(cf, test)
    "%.2f".format(eval.errorRate() * 100)
  }

}
