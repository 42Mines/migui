<?php

ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if(!isset($_FILES["photo"])){
	return ;
}

$photo = $_FILES["photo"];

// Testons si l'extension est autorisée
$infosfichier = pathinfo($photo['name']);
$extension_upload = $infosfichier['extension'];
$extensions_autorisees = array('jpg', 'jpeg');
if (in_array($extension_upload, $extensions_autorisees))
{
	$name = time();
	move_uploaded_file($photo['tmp_name'], 'uploads/' . "$name.$extension_upload");

	while(!file_exists(__DIR__. "/uploads/fire_$name.$extension_upload")){
		usleep(100);
	}

	echo $name;
}