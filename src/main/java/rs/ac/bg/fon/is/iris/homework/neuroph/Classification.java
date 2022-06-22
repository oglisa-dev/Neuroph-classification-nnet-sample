package rs.ac.bg.fon.is.iris.homework.neuroph;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author Ognjen Simic
 */
public class Classification {

    public static void main(String[] args) {
        System.out.println((new File("test")).getAbsolutePath());;
        String file_path = "./src/main/resources/wines.csv";
        int input_count = 13;
        int output_count = 3;
        System.out.println("Creating dataset...");
        DataSet dataSet = DataSet.createFromFile(file_path, input_count, output_count, ",");
        System.out.println("DataSet created...");

        System.out.println("Creating training and test datasets...");
        DataSet[] split = dataSet.split(0.7, 0.3);
        DataSet training_ds = split[0];
        DataSet test_ds = split[1];

        final ArrayList<MultiLayerPerceptron> nnets = new ArrayList<>(3);
        double[] lr_params = new double[]{0.2, 0.4, 0.5};
        final double MAX_ERROR = 0.02;
        Arrays.stream(lr_params).forEach(param -> {
            MultiLayerPerceptron nnet = new MultiLayerPerceptron(TransferFunctionType.TANH,
                    input_count, 22, output_count);
            nnets.add(nnet);
            MomentumBackpropagation mbp = new MomentumBackpropagation();
            nnet.setLearningRule(mbp);
            mbp.setLearningRate(param);
            mbp.setMaxError(MAX_ERROR);
            mbp.addListener((event) -> {
                MomentumBackpropagation lrule = (MomentumBackpropagation) event.getSource();
                if (event.getEventType() != LearningEvent.LEARNING_STOPPED) {
                    System.out.println(lrule.getCurrentIteration() + ". iteration[rate = " + param
                            + "] | " + "Total network error: " + lrule.getTotalNetworkError());
                } else {
                    System.out.println("Total number of iterations until finish: " + lrule.getCurrentIteration());
                }
            });
        });

        System.out.println("Training networks...");
        int sum = 0;
        for (MultiLayerPerceptron nnet : nnets) {
            nnet.learn(training_ds);
            MomentumBackpropagation lrule = (MomentumBackpropagation) nnet.getLearningRule();
            sum += lrule.getCurrentIteration();
        }
        System.out.println("Training completed...");
        System.out.println("Average number of iterations: " + sum / 3);
        
        // Evaluate all three nnets on create test dataset
        for(MultiLayerPerceptron nnet : nnets) {
            evaluate(nnet, test_ds);
        }
        
        // save neural networks to files
        System.out.println("Saving networks");
        for(int i=0;i<nnets.size();i++) {
            nnets.get(i).save("nn" + (i+1) + ".nnet");
        }
        
        System.out.println("Done.");
    }

    /**
     * Evaluates classification performance of a neural network. Contains
     * calculation of Confusion matrix for classification tasks or Mean Squared
     * Error (MSE) and Mean Absolute Error (MAE) for regression tasks.
     *
     * @param nnet
     * @param test_set
     */
    public static void evaluate(NeuralNetwork nnet, DataSet test_set) {
        System.out.println("Calculating performance indicators for neural network.");

        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));

        String[] classLabels = new String[]{"class1", "class2", "class3"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        
        evaluation.evaluate(nnet, test_set);
        
        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
        ConfusionMatrix result_cm = evaluator.getResult();
        System.out.println("Confusion matrrix:\r\n");
        System.out.println(result_cm + "\r\n\r\n");
        System.out.println("Classification metrics\r\n");
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(result_cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        for(ClassificationMetrics metric : metrics) {
            System.out.println(metric + "\r\n");
        }
        System.out.println(average.toString());
    }

}
