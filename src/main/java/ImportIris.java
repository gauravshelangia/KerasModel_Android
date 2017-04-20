import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by tomhanlon on 2/10/17.
 */
public class ImportIris {

    public static void main(String[] args) throws Exception {

        // Keras model saved was a sequential model
        // Use MultiLayerNetwork in Deeplearning4J when importing Sequential models
        // Use Computationgraph for keras models built using Functional API

        // Load the weights and config seperately
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("/tmp/iris_model_json", "/tmp/iris_model_weights");

        // Load the weights and config from single file
        MultiLayerNetwork model1 = KerasModelImport.importKerasSequentialModelAndWeights("/tmp/full_iris_model");


         // DeepLearning4j equivalent of keras model.to_json()
        //System.out.print(model.conf().toJson());


        // Our model expects input like this.
        // [ 7.2  3.   5.8  1.6]
        //4.6  3.6  1.   0.2
        //5.1  3.5  1.4  0.2
        //5.9  3.   5.1  1.8

        INDArray myArray = Nd4j.zeros(1, 4); // one row 4 column array
        myArray.putScalar(0,0, 4.6);
        myArray.putScalar(0,1, 3.6);
        myArray.putScalar(0,2, 1.0);
        myArray.putScalar(0,3, 0.2);

        INDArray output = model.output(myArray);
        System.out.println("First Model Output");
        System.out.println(myArray);
        System.out.println(output);

        INDArray output1 = model1.output(myArray);
        System.out.println("Second Model Output");
        System.out.println(myArray);
        System.out.println(output1);





    }
}