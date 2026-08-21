package com.brax.preductiontb;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private Module module;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        resultText = findViewById(R.id.resultText);

        try {
            // Charger le modèle .ptl depuis les assets
                module = Module.load(assetFilePath("best_model_mobile.ptl"));
            Toast.makeText(this, "Modèle chargé avec succès ✅", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Toast.makeText(this, "Erreur de chargement du modèle ❌", Toast.LENGTH_LONG).show();
            return;
        }

        // Exemple : charger une image depuis les ressources drawable
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test_image);

        // Convertir l'image en tenseur (normalisation ImageNet standard)
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );

        // Exécuter l'inférence
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // Récupérer les résultats
        float[] scores = outputTensor.getDataAsFloatArray();

        // Trouver la classe avec la plus grande probabilité
        int maxIndex = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[maxIndex]) maxIndex = i;
        }

        // Afficher le résultat
        resultText.setText("Classe prédite : " + maxIndex);
    }

    // Fonction utilitaire pour copier un fichier depuis assets vers un chemin lisible
    private String assetFilePath(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
        return file.getAbsolutePath();
    }
}
