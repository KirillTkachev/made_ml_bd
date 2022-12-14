package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset}


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

    val delta = 0.0001
    val numberOfIteration = 1000
    val stepSize = 0.6

    "Estimator" should "should produce functional model" in {
        val linearRegression = provideModel()
        val model = linearRegression.fit(LinearRegressionTest.data)
        validateModel(model)
    }

    "Estimator" should "work after re-read" in {
        val pipeline = new Pipeline().setStages(Array(
            provideModel()
        ))

        val tmpFolder = Files.createTempDir()
        pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
        val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
        val model = reRead.fit(LinearRegressionTest.data).stages(0).asInstanceOf[LinearRegressionModel]
        validateModel(model)
    }

    "Model" should "work after re-read" in {
        val pipeline = new Pipeline().setStages(Array(
            provideModel()
        ))

        val model = pipeline.fit(LinearRegressionTest.data)
        val tmpFolder = Files.createTempDir()
        model.write.overwrite().save(tmpFolder.getAbsolutePath)
        val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
        validateModel(reRead.stages(0).asInstanceOf[LinearRegressionModel])
    }

    private def provideModel() = {
        new LinearRegression()
            .setFeaturesCol("x")
            .setLabelCol("y")
            .setPredictionCol("prediction")
            .setNumberOfIterations(numberOfIteration)
            .setStepSize(stepSize)
    }

    private def validateModel(model: LinearRegressionModel) = {
        model.weights.size should be(3)
        model.weights(0) should be(1.5 +- delta)
        model.weights(1) should be(0.3 +- delta)
        model.weights(2) should be(-0.7 +- delta)
    }

    object LinearRegressionTest extends WithSpark {
        lazy val rowsNumber = 100000
        lazy val originalParameters: DenseVector[Double] = DenseVector[Double](1.5, 0.3, -0.7)

        lazy val X: DenseMatrix[Double] = DenseMatrix.rand[Double](rowsNumber, 3)
        lazy val y: DenseVector[Double] = {
            X * originalParameters
        }
        lazy val _data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

        lazy val _df: Dataset[_] = {
            import sqlc.implicits._
            _data(*, ::).iterator.map(x => {
                (x(0), x(1), x(2), x(3))
            }).toSeq.toDF("x1", "x2", "x3", "y")
        }

        lazy val assembler: VectorAssembler = new VectorAssembler()
            .setInputCols(Array("x1", "x2", "x3"))
            .setOutputCol("x")

       lazy val data: DataFrame = assembler.transform(_df).drop("x1", "x2", "x3")
    }
}