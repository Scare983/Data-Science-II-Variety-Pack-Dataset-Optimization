
import scalation.analytics.ActivationFun.f_sigmoid
import scalation.analytics.RegTechnique.QR
import scalation.analytics.TranRegression._
import scalation.math.sq

import scala.math.sqrt
import scalation.columnar_db.Relation
import scalation.linalgebra.{MatrixD, VectorD}
import scalation.plot.{Plot, PlotM}
import scalation.util.banner
import scalation.analytics.TranRegression

import scala.math.log
import scala.math.exp
import scalation.analytics.ActivationFun._
import scalation.analytics.Optimizer.hp
import scalation.analytics.{ELM_3L1, NeuralNet_3L, NeuralNet_XL, Optimizer, Perceptron, TranRegression}





/*object TranRegessionTest extends App{


  banner ("qsar_aquatic_toxicity relation")
  val auto_tab = Relation ("./DataSets/" + "qsar_aquatic_toxicity.csv", "qsar_aquatic_toxicity", null, -1)
  auto_tab.show()


  val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
  println (s"x = $x")
  println (s"y = $y")
  val _1 = VectorD.one (x.dim1)
  val t  = VectorD.range (0, x.dim2)
  val ox = _1 +^: x


  val f = (log _ , exp _)                                        // try several transforms
  // val f = (sqrt _ , sq _)
  // val f = (sq _ , sqrt _)
  TranRegression.setLambda (0.2)                                 // try 0.2, 0.3, 0.4, 0.5, 0.6
  //val f = (box_cox _ , cox_box _)
  TranRegression.rescaleOff ()

  banner (s"TranRegression with transform $f")
  val trg = TranRegression (ox, y, null, null, f._1, f._2, QR, null)    // automated
  println (trg.analyze ().report)
  println (trg.summary)

  banner ("Forward Selection Test")
  val (cols, rSq) = trg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
  val k = cols.size-1
  println (s"k = $k, n = ${x.dim2}")                         // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for TranRegression", lines = true)

  println (s"rSq = $rSq")

  println(f"The max value of rSq ::: ${rSq.col(0).max()}%1.5f")
  println(s"n* for max rSq ::: ${rSq.col(0).argmax()+1}")
  println(f"The max value of rSqBar ::: ${rSq.col(1).max()}%1.5f")
  println(s"n* for max rSqBar ::: ${rSq.col(1).argmax()+1}")
  println(f"The max value of rSqCV ::: ${rSq.col(2).max()}%1.5f")
  println(s"n* for max rSqCV ::: ${rSq.col(2).argmax()+1}")
}*/

object PerceptronTests extends App{
  banner ("Perceptron feature selection - QsarAquaticToxicity")
  val myVar  = Relation("../Data/boston.csv", "boston", null, -1)
  val auto_tab = Relation ("../Data/qsar_aquatic_toxicity.csv", "qsar_aquatic_toxicity", null, -1)
  auto_tab.show()

  val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
  val xy = auto_tab.toMatriD(0 to 8)
  println (s"x = $x")
  println (s"y = $y")
  val _1 = VectorD.one (x.dim1)
  //val t  = VectorD.range (0, x.dim2)
  val ox = _1 +^: x
  //val n = ox.dim2
  //val xy =
  val oxy = _1 +^: xy

  val f_ = f_sigmoid                                              // try different activation function
  //val f_ = f_tanh                                                 // try different activation function
  //val f_ = f_id                                                   // try different activation function
  //val f_ = f_reLU

  println ("ox = " + ox)
  println ("y  = " + y)

  banner ("Perceptron with scaled y values")
  val hp2 = Optimizer.hp.updateReturn (("eta", 0.8), ("bSize", 10.0),("maxEpochs",200))
  val nn  = Perceptron (oxy, f0 =  f_)                             // factory function automatically rescales
  //val nn  = new Perceptron (ox, y, f0 = f_)                       // constructor does not automatically rescale


  nn.train ().eval ()                                             // fit the weights using training data
  val n = ox.dim2                                                 // number of parameters/variables
  println (nn.report)

  banner ("Cross-Validation Test")
  nn.crossValidate ()

  banner ("Forward Selection Test")
  val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
  println (s"rSq = $rSq")
  val k = cols.size
  println (s"k = $k, n = $n")
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for Perceptron", lines = true)

  println(f"The max value of rSq ::: ${rSq.col(0).max()}%1.5f")
  println(s"n* for max rSq ::: ${rSq.col(0).argmax()+1}")
  println(f"The max value of rSqBar ::: ${rSq.col(1).max()}%1.5f")
  println(s"n* for max rSqBar ::: ${rSq.col(1).argmax()+1}")
  println(f"The max value of rSqCV ::: ${rSq.col(2).max()}%1.5f")
  println(s"n* for max rSqCV ::: ${rSq.col(2).argmax()+1}")

  val epochs = VectorD.range (1,  hp2("maxEpochs").toInt)
  println(epochs)



}

object NeuralNetwork3LTests extends App {

  banner ("NeuralNetwork3XL feature selection - QsarAquaticToxicity")
  val auto_tab = Relation ("../Data/" + "qsar_aquatic_toxicity.csv", "qsar_aquatic_toxicity", null, 0)
  auto_tab.show()

  val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
  val xy = auto_tab.toMatriD(0 to 8)
  println (s"x = $x")
  println (s"y = $y")
  val _1 = VectorD.one (x.dim1)
  //val t  = VectorD.range (0, x.dim2)
  val ox = _1 +^: x
  val n = ox.dim2
  //val xy =
  val oxy = _1 +^: xy



  val f_ = (f_sigmoid, f_id)                                     // try different activation functions
  //  val f_ = (f_tanh, f_id)                                        // try different activation functions
  //  val f_ = (f_lreLU, f_id)                                       // try different activation functions

  banner ("NeuralNet_3L with scaled y values")
  hp("eta") = 0.02
  hp("bSize") = 10.0
  hp("maxEpochs") = 200

  val nn  = NeuralNet_3L (oxy, f0 = f_._1, f1 = f_._2)           // factory function automatically rescales

  nn.train ().eval ()                                            // fit the weights using training data
  println (nn.report)                                            // report parameters and fit
  val ft  = nn.fitA(0)                                           // fit for first output variable

  banner ("Forward Selection Test")
  val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
  println (s"rSq = $rSq")
  val k = cols.size
  //println (s"k = $k, n = $n")
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_3L", lines = true)

  println(f"The max value of rSq ::: ${rSq.col(0).max()}%1.5f")
  println(s"n* for max rSq ::: ${rSq.col(0).argmax()+1}")
  println(f"The max value of rSqBar ::: ${rSq.col(1).max()}%1.5f")
  println(s"n* for max rSqBar ::: ${rSq.col(1).argmax()+1}")
  println(f"The max value of rSqCV ::: ${rSq.col(2).max()}%1.5f")
  println(s"n* for max rSqCV ::: ${rSq.col(2).argmax()+1}")
}

object NeuralNetworkXLTests extends App {
  banner ("NeuralNetwork3XL feature selection - QsarAquaticToxicity")
  val auto_tab = Relation ("./DataSets/" + "qsar_aquatic_toxicity.csv", "qsar_aquatic_toxicity", null, -1)
  auto_tab.show()

  val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
  val xy = auto_tab.toMatriD(0 to 8)
  println (s"x = $x")
  println (s"y = $y")
  val _1 = VectorD.one (x.dim1)
  //val t  = VectorD.range (0, x.dim2)
  val ox = _1 +^: x

  //val xy =
  val oxy = _1 +^: xy

  val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions

  banner ("NeuralNet_XL with scaled y values")
  hp("eta") = 0.02                                               // learning rate hyper-parameter (see Optimizer)
  val nn  = NeuralNet_XL (oxy, af = af_)                         // factory function automatically rescales

  nn.train ().eval ()                                            // fit the weights using training data
  println (nn.report)                                            // report parameters and fit
  val ft  = nn.fitA(0)                                           // fit for first output variable

  banner ("Forward Selection Test")
  val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
  println (s"rSq = $rSq")
  val k = cols.size
  //println (s"k = $k, n = $n")
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for NeuralNet_XL", lines = true)

  println(f"The max value of rSq ::: ${rSq.col(0).max()}%1.5f")
  println(s"n* for max rSq ::: ${rSq.col(0).argmax()+1}")
  println(f"The max value of rSqBar ::: ${rSq.col(1).max()}%1.5f")
  println(s"n* for max rSqBar ::: ${rSq.col(1).argmax()+1}")
  println(f"The max value of rSqCV ::: ${rSq.col(2).max()}%1.5f")
  println(s"n* for max rSqCV ::: ${rSq.col(2).argmax()+1}")
}

object ExtremeLearningMachineTests extends App{

  banner ("NeuralNetwork3XL feature selection - QsarAquaticToxicity")
  val auto_tab = Relation ("./DataSets/" + "qsar_aquatic_toxicity.csv", "qsar_aquatic_toxicity", null, -1)
  auto_tab.show()

  val (x, y) = auto_tab.toMatriDD (0 to 7, 8)
  val xy = auto_tab.toMatriD(0 to 8)
  println (s"x = $x")
  println (s"y = $y")
  val _1 = VectorD.one (x.dim1)
  //val t  = VectorD.range (0, x.dim2)
  val ox = _1 +^: x

  //val xy =
  val oxy = _1 +^: xy

  val f_ = f_tanh                                                 // try different activation functions

  banner ("ELM_3L1 with scaled y values")
  val nn  = ELM_3L1 (oxy, f0 = f_)                                // factory function automatically rescales

  nn.train ().eval ()                                             // fit the weights using training data
  println (nn.report)                                             // report parameters and fit

  banner ("Forward Selection Test")
  val (cols, rSq) = nn.forwardSelAll ()                           // R^2, R^2 bar, R^2 cv
  println (s"rSq = $rSq")
  val k = cols.size
  //println (s"k = $k, n = $n")
  val t = VectorD.range (1, k)                                    // instance index
  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for ELM_3L1", lines = true)

  println(f"The max value of rSq ::: ${rSq.col(0).max()}%1.5f")
  println(s"n* for max rSq ::: ${rSq.col(0).argmax()+1}")
  println(f"The max value of rSqBar ::: ${rSq.col(1).max()}%1.5f")
  println(s"n* for max rSqBar ::: ${rSq.col(1).argmax()+1}")
  println(f"The max value of rSqCV ::: ${rSq.col(2).max()}%1.5f")
  println(s"n* for max rSqCV ::: ${rSq.col(2).argmax()+1}")
}



