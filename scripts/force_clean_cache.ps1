$path = "C:\Users\Xuan\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5"
if (Test-Path $path) {
    Write-Host "Found cache directory: $path"
    try {
        # Force remove the item and all children
        Remove-Item -Path $path -Recurse -Force -ErrorAction Stop
        Write-Host "Successfully deleted cache directory."
    } catch {
        Write-Error "Failed to delete directory. Error: $_"
        Write-Host "Attempting to reset permissions and try again..."
        
        # Try to take ownership and reset permissions if simple delete fails
        takeown /f $path /r /d y
        icacls $path /grant administrators:F /t
        
        try {
            Remove-Item -Path $path -Recurse -Force -ErrorAction Stop
            Write-Host "Successfully deleted cache directory after permission reset."
        } catch {
            Write-Error "Still unable to delete. Please delete manually."
        }
    }
} else {
    Write-Host "Cache directory not found. It might have been already deleted."
}
