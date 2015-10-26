package project;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.Arrays;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


        for (int i = 0; i < 3 ; i++) {
            for (int baseIteration : new int[]{1000, 10000, 50000, 100000, 150000, 200000}) {

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
