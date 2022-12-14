package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends HasLabelCol with HasFeaturesCol with HasPredictionCol
    with HasStepSize {

    def setPredictionCol(value: String): this.type = set(predictionCol, value)

    def setFeaturesCol(value: String): this.type = set(featuresCol, value)

    def setLabelCol(value: String): this.type = set(labelCol, value)

    def setStepSize(value: Double): this.type = set(stepSize, value)

    val numberOfIterations: IntParam = new IntParam(this, "numberOfIterations", "Number of iterations")

    def setNumberOfIterations(value: Int): this.type = set(numberOfIterations, value)

    setDefault(numberOfIterations -> 100000, stepSize -> 0.1)

    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

        if (schema.fieldNames.contains($(labelCol))) {
            SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)
        } else {
            SchemaUtils.appendColumn(schema, StructField(getLabelCol, DoubleType))
        }

        if (schema.fieldNames.contains($(predictionCol))) {
            SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
        } else {
            SchemaUtils.appendColumn(schema, StructField(getPredictionCol, new VectorUDT()))
        }

        schema
    }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
    with DefaultParamsWritable {

    def this() = this(Identifiable.randomUID("linearRegression_"))

    def backPropagation(data: Vector, weights: breeze.linalg.DenseVector[Double]): Vector = {
        val X = data.asBreeze(0 until weights.size).toDenseVector
        val y = data.asBreeze(-1)
        val grads = X * (breeze.linalg.sum(X * weights) - y)
        Vectors.fromBreeze(grads)
    }

    override def fit(dataset: Dataset[_]): LinearRegressionModel = {
        implicit val encoder: Encoder[Vector] = ExpressionEncoder()
        val assembler = new VectorAssembler()
            .setInputCols(Array("b", $(featuresCol), $(labelCol)))
            .setOutputCol("features")

        val features = assembler.transform(dataset.withColumn("b", lit(1))).select("features").as[Vector]
        val featuresSize = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
        var weights: breeze.linalg.DenseVector[Double] = breeze.linalg.DenseVector.zeros(featuresSize + 1)

        for (_ <- 0 to $(numberOfIterations)) {
            val summary = features.rdd.mapPartitions((data: Iterator[Vector]) => {
                val summarizer = new MultivariateOnlineSummarizer()
                data.foreach(vector => {
                    summarizer.add(mllib.linalg.Vectors.fromBreeze(backPropagation(vector, weights).asBreeze))
                })
                Iterator(summarizer)
            }).reduce(_ merge _)

            weights = weights - $(stepSize) * summary.mean.asBreeze
        }

        copyValues(new LinearRegressionModel(
            Vectors.fromBreeze(weights(1 until weights.size)).toDense, weights(0))).setParent(this)

    }

    override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                             override val uid: String,
                                             val weights: Vector,
                                             val b: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {
    private[made] def this(weights: Vector, b: Double) =
        this(Identifiable.randomUID("linearRegressionModel_"), weights, b)

    override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
        new LinearRegressionModel(weights, b), extra)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val bWeights = weights.asBreeze
        val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
            (x: Vector) => {
                Vectors.fromBreeze(breeze.linalg.DenseVector(bWeights.dot(x.asBreeze)) + b)
            })
        dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
    }

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def write: MLWriter = new DefaultParamsWriter(this) {
        override protected def saveImpl(path: String): Unit = {
            super.saveImpl(path)
            val wrappedWeights = weights -> b
            sqlContext.createDataFrame(Seq(wrappedWeights)).write.parquet(path + "/weights")
        }
    }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
    override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
        override def load(path: String): LinearRegressionModel = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc)

            val wrappedWeights = sqlContext.read.parquet(path + "/weights")

            implicit val encoder: Encoder[Vector] = ExpressionEncoder()

            val weights = wrappedWeights.select(wrappedWeights("_1").as[Vector]).first()
            val b = wrappedWeights.select(wrappedWeights("_2")).first().getDouble(0)
            val model = new LinearRegressionModel(weights, b)
            metadata.getAndSetParams(model)
            model
        }
    }
}