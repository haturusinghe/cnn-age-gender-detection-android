package fct.cs.cnn_cam_gender

import android.app.ProgressDialog
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.media.ExifInterface
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.navigation.ui.AppBarConfiguration
import com.google.android.material.snackbar.Snackbar
import fct.cs.cnn_cam_gender.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.IOException


class MainActivity : AppCompatActivity() {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding

    private lateinit var sampleImageView : ImageView

    private val shift = 5

    private var useNNApi : Boolean = false
    private var useGpu : Boolean = false

    private val coroutineScope = CoroutineScope( Dispatchers.Main )

    lateinit var genderModelInterpreter: Interpreter
    private lateinit var genderClassificationModel: GenderClassificationModel

    private val compatList = CompatibilityList()

    private var modelFilename = arrayOf( "model_age_q.tflite", "model_gender_q.tflite" )

    // For reading the full-sized picture
    private val REQUEST_IMAGE_CAPTURE = 101
    private val REQUEST_IMAGE_SELECT = 102
    private lateinit var currentPhotoPath : String

    private lateinit var progressDialog : ProgressDialog

    override fun onCreate(savedInstanceState: Bundle?)  {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

//        val navController = findNavController(R.id.nav_host_fragment_content_main)
//        appBarConfiguration = AppBarConfiguration(navController.graph)
//        setupActionBarWithNavController(navController, appBarConfiguration)

        binding.fab.setOnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show()
            openCamera(view)
        }

        binding.btnSelectImg.setOnClickListener { view ->
            selectImage(view)
        }

        // A ProgressDialog to notify the user that the images are being processed.
        progressDialog = ProgressDialog( this )
        progressDialog.setCancelable( false )
        progressDialog.setMessage( "Processing ...")

        val useNNApiCheckBox : CheckBox = (binding.useNNApiCheckbox)
        val useGPUCheckBox : CheckBox = (binding.useGPUCheckbox)
        val initModelButton : Button = (binding.initModelButton)
        sampleImageView = binding.iv

        // Check for NNAPI and GPUDelegate compatibility.
        if ( Build.VERSION.SDK_INT < Build.VERSION_CODES.P ) {
            useNNApiCheckBox.isEnabled = false
            useNNApiCheckBox.text = "Use NNAPI ( NNAPI is not available on this Device)"
            useNNApi = false
        }
        if ( !compatList.isDelegateSupportedOnThisDevice ){
            useGPUCheckBox.isEnabled = false
            useGPUCheckBox.text = "Use GPU ( GPU acceleration is not available on this device )."
            useGpu = false
        }

        useNNApiCheckBox.setOnCheckedChangeListener { buttonView, isChecked ->
            useNNApi = isChecked
        }
        useGPUCheckBox.setOnCheckedChangeListener { buttonView, isChecked ->
            useGpu = isChecked
        }

        initModelButton.setOnClickListener {
            val options = Interpreter.Options().apply {
                if ( useGpu ) {
                    addDelegate(GpuDelegate( compatList.bestOptionsForThisDevice ) )
                }
                if ( useNNApi ) {
                    addDelegate(NnApiDelegate())
                }
            }
            // Initialize the models in a coroutine.
            coroutineScope.launch {
                initModels(options)
            }
        }
    }

    // `onClick` method for R.id.button
    fun openCamera( v: View) {
        dispatchTakePictureIntent()
    }

    // `onClick` method for R.id.button2
    fun selectImage( v : View) {
        dispatchSelectPictureIntent()
    }

/*    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration)
                || super.onSupportNavigateUp()
    }*/

    // Suspending function to initialize the TFLite interpreters.
    private suspend fun initModels(options: Interpreter.Options) = withContext( Dispatchers.Default ) {
        genderModelInterpreter = Interpreter(FileUtil.loadMappedFile( applicationContext , modelFilename[1]), options )
        withContext( Dispatchers.Main ){
            genderClassificationModel = GenderClassificationModel().apply {
                interpreter = genderModelInterpreter
            }
            // Notify the user once the models have been initialized.
            Toast.makeText( applicationContext , "Models initialized." , Toast.LENGTH_LONG ).show()
        }
    }

    private fun cropToBBox(image: Bitmap, bbox: Rect) : Bitmap {
        return Bitmap.createBitmap(
            image,
            bbox.left - 0 * shift,
            bbox.top + shift,
            bbox.width() + 0 * shift,
            bbox.height() + 0 * shift
        )
    }


    // Create a temporary file, for storing the full-sized picture taken by the user.
    private fun createImageFile() : File {
        val imagesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("image", ".jpg", imagesDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    // Dispatch an Intent which opens the gallery application for the user.
    private fun dispatchSelectPictureIntent() {
        val selectPictureIntent = Intent( Intent.ACTION_OPEN_DOCUMENT ).apply {
            type = "image/*"
            addCategory( Intent.CATEGORY_OPENABLE )
        }
        startActivityForResult( selectPictureIntent , REQUEST_IMAGE_SELECT )
    }

    // Dispatch an Intent which opens the camera application for the user.
    // The code is from -> https://developer.android.com/training/camera/photobasics#TaskPath
    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent( MediaStore.ACTION_IMAGE_CAPTURE )
        if ( takePictureIntent.resolveActivity( packageManager ) != null ) {
            val photoFile: File? = try {
                createImageFile()
            }
            catch (ex: IOException) {
                null
            }
            photoFile?.also {
                val photoURI = FileProvider.getUriForFile(
                    this,
                    "fct.cs.cnn_cam_gender", it
                )
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }


    private fun rotateBitmap(original: Bitmap, degrees: Float): Bitmap? {
        val matrix = Matrix()
        matrix.preRotate(degrees)
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        // If the user opened the camera
        if ( resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_CAPTURE ) {
            // Get the full-sized Bitmap from `currentPhotoPath`.
            var bitmap = BitmapFactory.decodeFile( currentPhotoPath )
            val exifInterface = ExifInterface( currentPhotoPath )
            bitmap =
                when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION , ExifInterface.ORIENTATION_UNDEFINED )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap( bitmap , 90f )
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap( bitmap , 180f )
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap( bitmap , 270f )
                    else -> bitmap
                }
            progressDialog.show()
            // Pass the clicked picture to `detectFaces`.
            detectFaces( bitmap!! )
        }
        // if the user selected an image from the gallery
        else if ( resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_SELECT ) {
            val inputStream = contentResolver.openInputStream( data?.data!! )
            val bitmap = BitmapFactory.decodeStream( inputStream )
            inputStream?.close()
            progressDialog.show()
            // Pass the clicked picture to `detectFaces`.
            detectFaces( bitmap!! )
        }
    }
    private fun detectFaces(image: Bitmap) {
        sampleImageView.setImageBitmap(image);
    }

}