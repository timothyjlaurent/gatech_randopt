package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
         int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


        for (int baseIteration : new int[] { 1000, 10000, 50000, 100000, 150000, 200000 }) {

            long startTime = System.currentTimeMillis();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, baseIteration);
            fit.train();
            long endTime = System.currentTimeMillis();
            System.out.println("RHC\t" + baseIteration + "\t" +  (endTime - startTime) + "\t" +  ef.value(rhc.getOptimal()));

            startTime = System.currentTimeMillis();
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, baseIteration);
            fit.train();
            endTime = System.currentTimeMillis();
            System.out.println("SA\t" + baseIteration + "\t" + (endTime - startTime) + "\t" + ef.value(sa.getOptimal()));

            startTime = System.currentTimeMillis();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(100, 75, 12, gap);
            fit = new FixedIterationTrainer(ga, baseIteration/100);
            fit.train();
            endTime = System.currentTimeMillis();
            System.out.println("GA\t" + baseIteration + "\t" + (endTime - startTime) + "\t" + ef.value(ga.getOptimal()));

            startTime = System.currentTimeMillis();
            MIMIC mimic = new MIMIC(100, 50, pop);
            fit = new FixedIterationTrainer(mimic,  baseIteration/100);
            fit.train();
            endTime = System.currentTimeMillis();
            System.out.println("MIMIC\t" + baseIteration + "\t" + (endTime - startTime) + "\t" + ef.value(mimic.getOptimal()));
        }
    }

}
