package shared.tester;

import shared.Instance;

/**
 * Created by timothylaurent on 10/25/15.
 */

public class AccuracyTestMetricCategorical extends TestMetric {

    private int count;
    private int countCorrect;

    @Override
    public void addResult(Instance expected, Instance actual) {
        count++;
        if (isAllCorrect(expected, actual)) {
            countCorrect++;
        }
    }

    public double getPctCorrect() {
        return count > 0 ? ((double)countCorrect)/count : 1; //if count is 0, we consider it all correct
    }

    public void printResults() {
        //only report results if there were any results to report.
        if (count > 0) {
            double pctCorrect = getPctCorrect();
            double pctIncorrect = (1 - pctCorrect);
            System.out.print(String.format("%.05f\n",   100 * pctCorrect));
//            System.out.println(String.format("Incorrectly Classified Instances: %.02f%%", 100 * pctIncorrect));
        } else {

            System.out.println("No results added.");
        }
    }

    public boolean isAllCorrect(Instance expected, Instance actual) {
        // Check feature values.
        for (int i = 0; i < expected.size(); i++) {
            if ( Math.round(expected.getContinuous(i)) != Math.round(actual.getContinuous(i))) {
                return false;
            }
        }
        return true;
    }
}
