package project;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.example.TravelingSalesmanSortEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.Arrays;
import java.util.Random;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        for(int i = 0 ; i < 3 ; i++) {
            for (int baseIteration : new int[]{100, 500, 800, 1000, 2000, 3000, 5000, 7000, 10000, 50000}) {

                long startTime = System.currentTimeMillis();
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, baseIteration);
                fit.train();
                long endTime = System.currentTimeMillis();
                System.out.println("RHC\t" + baseIteration + "\t" + (endTime - startTime) + "\t" + ef.value(rhc.getOptimal()));

                startTime = System.currentTimeMillis();
                SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
                fit = new FixedIterationTrainer(sa, baseIteration);
                fit.train();
                endTime = System.currentTimeMillis();
                System.out.println("SA\t" + baseIteration + "\t" + (endTime - startTime) + "\t" + ef.value(sa.getOptimal()));

                startTime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(100, 75, 12, gap);
                fit = new FixedIterationTrainer(ga, baseIteration / 100);
                fit.train();
                endTime = System.currentTimeMillis();
                System.out.println("GA\t" + baseIteration + "\t" + (endTime - startTime) + "\t" + ef.value(ga.getOptimal()));

                // for mimic we use a sort encoding
                ef = new TravelingSalesmanSortEvaluationFunction(points);
                int[] ranges = new int[N];
                Arrays.fill(ranges, N);
                odd = new DiscreteUniformDistribution(ranges);
                Distribution df = new DiscreteDependencyTree(.1, ranges);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                startTime = System.currentTimeMillis();
                MIMIC mimic = new MIMIC(100, 50, pop);
                fit = new FixedIterationTrainer(mimic, baseIteration / 100);
                fit.train();
                endTime = System.currentTimeMillis();
                System.out.println("MIMIC\t" + baseIteration + "\t" + (endTime - startTime) + "\t" + ef.value(mimic.getOptimal()));
            }
        }
    }
}
