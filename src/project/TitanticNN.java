package project;

import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import shared.filt.LabelSplitFilter;
import shared.reader.CSVDataSetReader;
import shared.reader.DataSetReader;
import shared.tester.*;
import util.linalg.DenseVector;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;



/**
 * Created by timothylaurent on 10/24/15.
 */
public class TitanticNN {
    /**
     * The test main
     * @param args ignored parameters
     */
    public static void main(String[] args) throws Exception {
        double[][][] datatable;
        DataSetReader dsr = new CSVDataSetReader(new File("").getAbsolutePath() + "/data/titantic_data_bin_norm.csv");
        // read in the raw data
        DataSet ds = dsr.read();
        System.out.println(ds);

        // split out the label
        LabelSplitFilter lsf = new LabelSplitFilter(1);
        lsf.filter(ds);
        System.out.println(ds);
        for (int i = 0 ; i < 3; i += 1 ) {
            run_nn_rhc(ds, new int[]{100, 500, 1000, 5000, 10000, 15000, 20000}, "RHC");
            run_nn_rhc(ds, new int[]{1, 5, 10, 50, 100, 150, 200}, "GA");
            run_nn_rhc(ds, new int[]{100, 500, 1000, 5000, 10000, 15000, 20000}, "SA");
        }

    }

    public static double[][][] makeArTable(DataSet ds){
        double[][][] datatable;
        double[][][] temptable;

        ArrayList<double[][]> dataarlist = new ArrayList<double[][]>();
        for (Instance inst :  ds.getInstances() ) {
            System.out.println(inst);
            double[][] row = makeArRow(inst, 10);
            System.out.println(row);
            dataarlist.add(row);
        }

        datatable = new double[dataarlist.size()][][];

        int i = 0;
        for (double[][] pair: dataarlist) {
            datatable[i] = pair;
            i++;
        }

        return datatable;
    }


    public static double[][] makeArRow(Instance inst, int labelCol) {

        DenseVector x1 = (DenseVector)inst.getData().get(0,labelCol);
        double[] x1_ar = x1.data;
        DenseVector x2 = (DenseVector)inst.getData().get(labelCol+1, inst.getData().size());
        double[] x2_ar = x2.data;

        DenseVector y = (DenseVector)inst.getData().get(labelCol, labelCol+1);
        double[] y_ar = y.data;
        // todo this does not handle the general case.
        double[][] out = {x1_ar, y_ar};
        return out;
    }

    public static void run_nn_rhc(DataSet set, int[] num_iters, String optAlgo ){

        for (int num : num_iters) {
            run_nn_rhc(set, num, optAlgo);
        }
    }


    public static FeedForwardNetwork run_nn_rhc(DataSet set, int num_iters, String optAlgo){
        FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
        // 2a) These numbers correspond to the number of nodes in each layer.
        //     This network has 4 input nodes, 3 hidden nodes in 1 layer, and 1 output node in the output layer.
        FeedForwardNetwork network = factory.createClassificationNetwork(new int[]{10, 10, 10,1});

        // 3) Instantiate a measure, which is used to evaluate each possible set of weights.
        ErrorMeasure measure = new SumOfSquaresError();

        // 5) Instantiate an optimization problem, which is used to specify the dataset, evaluation
        //    function, mutator and crossover function (for Genetic Algorithms), and any other
        //    parameters used in optimization.
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(
                set, network, measure);
        OptimizationAlgorithm o;
        switch(optAlgo) {
            case "RHC": o = new RandomizedHillClimbing(nno);
                        break;
            case "GA": o = new StandardGeneticAlgorithm(100, 10, 40, nno);
                        break;
            case "SA": o = new SimulatedAnnealing(5000, 0.999, nno);
                        break;
            default: o = new RandomizedHillClimbing(nno);
                    break;
        }
//        OptimizationAlgorithm o = new RandomizedHillClimbing(nno);

        // 7) Instantiate a trainer.  The FixtIterationTrainer takes another trainer (in this case,
        //    an OptimizationAlgorithm) and executes it a specified number of times.
        FixedIterationTrainer fit = new FixedIterationTrainer(o, num_iters);

        long startTime = System.currentTimeMillis();
        // 8) Run the trainer.  This may take a little while to run, depending on the OptimizationAlgorithm,
        //    size of the data, and number of iterations.
        fit.train();

        long endTime = System.currentTimeMillis();

        // 9) Once training is done, get the optimal solution from the OptimizationAlgorithm.  These are the
        //    optimal weights found for this network.
        Instance opt = o.getOptimal();
        network.setWeights(opt.getData());
        //10) Run the training data through the network with the weights discovered through optimization, and
        //    print out the expected label and result of the classifier for each instance.
        TestMetric acc = new AccuracyTestMetricCategorical();
//        TestMetric cm  = new ConfusionMatrixTestMetric(new int[] {0,1});
//        Tester t = new NeuralNetworkTester(network, acc, cm);
        Tester t = new NeuralNetworkTester(network, acc);
        t.test(set.getInstances());
//
        System.out.print( optAlgo + "\t" + num_iters + "\t" + (endTime - startTime) + "\t");

        acc.printResults();
//        cm.printResults();

        return network;


    }



}