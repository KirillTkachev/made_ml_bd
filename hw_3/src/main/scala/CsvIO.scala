package org.hw3.made

import breeze.linalg.{DenseMatrix, DenseVector, csvread, csvwrite}

import java.io.File


object CsvIO {

  def readCsv(path: String): DenseMatrix[Double] = {
    csvread(new File(path: String))
  }

  def writeCsv(path: String, data: DenseMatrix[Double]): Unit = {
    csvwrite(new File(path: String), data)
  }

}
