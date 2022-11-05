package org.hw3.made

import breeze.linalg.{*, DenseMatrix, DenseVector}

import java.io.PrintWriter


object Main {
  def main(args: Array[String]): Unit = {

    val trainData: DenseMatrix[Double] = CsvIO.readCsv(args(0))
    val testData: DenseMatrix[Double] = CsvIO.readCsv(args(1))

    val resultPath: String = args(2)
    val logsPath: String = args(3)

    val trainColumns: Int = trainData.cols
    val trainRows: Int = trainData.rows

    val testColumns: Int = testData.cols
    val testRows: Int = testData.rows

    val xTest: DenseMatrix[Double] = testData(0 until testRows, 0 until testColumns - 1)
    val yTest: DenseVector[Double] = testData(::, testColumns - 1)

    val xTrain: DenseMatrix[Double] = trainData(0 until trainRows, 0 until trainColumns - 1)
    val yTrain: DenseVector[Double] = trainData(::, trainColumns - 1)

    val logger: PrintWriter = new PrintWriter(logsPath)
    val model = new LinearRegression(logger)

    model.fit(xTrain, yTrain, xTest, yTest, numberOfIterations = 100_000)

    val prediction = model.predict(xTest)

    CsvIO.writeCsv(resultPath, prediction.asDenseMatrix.t)
  }
}
