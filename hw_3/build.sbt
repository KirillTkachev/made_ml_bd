ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "2.1.0"
)

lazy val root = (project in file("."))
  .settings(
    name := "hw_3",
    idePackagePrefix := Some("org.hw3.made")
  )
