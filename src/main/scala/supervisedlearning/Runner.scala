package supervisedlearning

import java.io.{FileInputStream, PrintWriter}

import weka.core.Instances
import weka.core.converters.ConverterUtils.DataSource


object Runner extends App {

  // TODO fix hackish java style
  var segmentChallengeInstances: Instances = null
  var zooInstances: Instances = null

  println("Kefu Zhou, CS 4641 Project 1, Supervised Learning")
  val pw = new PrintWriter(s"./result${System.currentTimeMillis()/100000L}.txt")
  readInstances()
  pw.write("*****Dataset: segment-challenge.arff*****\n")
  runDataset(segmentChallengeInstances)
  pw.write("\n")
  pw.write("*****Dataset: zoo.arff*****\n")
  runDataset(zooInstances)
  pw.write("\n")
  pw.close()


  private def readInstances() {
    val tmp1: FileInputStream = new FileInputStream("./data/segment-challenge.arff")
    segmentChallengeInstances = DataSource.read(tmp1)
    tmp1.close()
    val tmp2: FileInputStream = new FileInputStream("./data/zoo.arff")
    zooInstances = DataSource.read(tmp2)
    tmp2.close()
    if (segmentChallengeInstances.classIndex < 0) {
      segmentChallengeInstances.setClassIndex(segmentChallengeInstances.numAttributes - 1)
    }
    if (zooInstances.classIndex < 0) {
      zooInstances.setClassIndex(zooInstances.numAttributes - 1)
    }
  }

  private def runDataset(ins: Instances) {
    DecisionTree.run(pw, ins)
    Boosting.run(pw, ins)
    NeuralNetwork.run(pw, ins)
    KNN.run(pw, ins)
    SVM_Poly.run(pw, ins)
    SVM_RBF.run(pw, ins)
    pw.println()
  }

  def outputStringMatrix(mx: Array[Array[String]]) {
    mx.foreach {
      row =>
        (0 until row.length).foreach {
          j =>
            pw.append(row(j).trim)
            pw.append("\t")
        }
        pw.println()
    }
    pw.println()
  }

}
