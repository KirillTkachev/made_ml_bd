package org.hw3.made

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}

import java.io.PrintWriter

class LinearRegression(val printWriter: PrintWriter, val learningRate: Double = 0.1) {

  private var w: DenseVector[Double] = DenseVector.fill(1)(0)

  def fit(xTrain: DenseMatrix[Double], yTrain: DenseVector[Double], xTest: DenseMatrix[Double],
          yTest: DenseVector[Double], numberOfIterations: Int): Unit = {
    w = DenseVector.fill(xTrain.cols)(0)

    for (i <- 0 to numberOfIterations) {
      val predictions = predict(xTrain)
      val grad = 2.0 * xTrain.t * (predictions - yTrain) / (yTrain.length: Double)
      w = w - learningRate * grad
      if ((i + 1) % 1000 == 0) {
        val prediction = predict(xTest)
        val mse = getCurrentMseLoss(prediction, yTest)
        printWriter.write("\nIteration = " + i + " mse loss = " + mse)
      }
    }
    printWriter.close()
  }

  def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
    x * w
  }

  def getCurrentMseLoss(prediction: DenseVector[Double], trueValues: DenseVector[Double]): Double = {
    val mse = ((prediction - trueValues) * (prediction - trueValues)) / (trueValues.length: Double)
    sum(mse)
  }
}
