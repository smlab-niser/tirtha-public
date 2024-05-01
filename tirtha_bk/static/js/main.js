// Elements to be manipulated
const doc = document;
const body = doc.querySelector("body");
const nav = doc.querySelector("nav");
const side = doc.getElementById("side");
const modelArea = doc.getElementById("model-area");
const model = doc.getElementById("model");
const fControls = doc.getElementById("floating-controls");
const infoBtn = doc.getElementById("info-btn");
const PRE_URL = ""; // TODO: Use env

// ========================== FS START ==========================
// ❗ Handle fullscreen❗
const expandBtn = doc.getElementById("expand");
const expParent = expandBtn.parentElement;

function isInFullScreen() {
  return (
    (doc.fullScreenElement && doc.fullScreenElement !== null) ||
    doc.mozFullScreen ||
    doc.webkitIsFullScreen
  );
}

function requestFullScreen() {
  el = modelArea;
  var requestMethod =
    el.requestFullScreen ||
    el.webkitRequestFullScreen ||
    el.mozRequestFullScreen ||
    el.msRequestFullScreen;
  requestMethod.call(el);

  setTimeout(() => {
    model.classList.add("model-fs");
    doc.querySelector(".controls").style.display = "none";
  }, 100);
}

function exitFullScreen() {
  el = doc;
  var requestMethod =
    el.cancelFullScreen ||
    el.webkitCancelFullScreen ||
    el.mozCancelFullScreen ||
    el.exitFullscreen ||
    el.webkitExitFullscreen;
  requestMethod.call(el);

  setTimeout(() => {
    model.classList.remove("model-fs");
    doc.querySelector(".controls").style.display = "flex";
  }, 100);
}

expandBtn.addEventListener("click", () => {
  if (!isInFullScreen()) {
    setTimeout(() => {
      expParent.style.top = "0.5rem";
    }, 100);
    requestFullScreen();
  } else {
    setTimeout(() => {
      expParent.style.top = "0";
    }, 100);
    exitFullScreen();
  }
});

// Exit when fullscreen is exited via Esc keypress
function exitIfFS() {
  if (!isInFullScreen()) {
    setTimeout(() => {
      expParent.style.top = "0";
      model.classList.remove("model-fs");
      doc.querySelector(".controls").style.display = "flex";
    }, 100);
  }
}

doc.addEventListener("fullscreenchange", exitIfFS);
doc.addEventListener("webkitfullscreenchange", exitIfFS);
doc.addEventListener("mozfullscreenchange", exitIfFS);
doc.addEventListener("MSFullscreenChange", exitIfFS);
// ========================== FS END ==========================

// ========================== SCROLL DOWN START ==========================
let timeoutId;
function debounce(func, delay) {
  clearTimeout(timeoutId);
  timeoutId = setTimeout(func, delay);
}

// This fixes the show / hide info getting stuck prob
// And the nav bar getting stuck if opened and window resized
window.addEventListener("resize", () => {
  if (screen.width > 768) {
    body.style.setProperty(
      "--side-current-width",
      side.clientWidth + 15 + "px"
    );
    debounce(contDialog.close(), 100); // Close modal
    debounce(
      setTimeout(() => {
        // Close nav
        if (window.innerWidth < 768) {
          side.classList.remove("hide-side");
          modelArea.classList.remove("hide-side-model-area");
          doc.querySelectorAll("#info-btn > span")[1].textContent =
            "Show information";
        }
        nav.classList.remove("translate-nav");
        fControls.classList.remove("translate-floating-controls");
        body.classList.remove("blur");
        body.classList.remove("overflow-toggle");
      }, 100),
      100
    );
  }
});
// }
// ========================== SCROLL DOWN END ==========================

// ========================== NAV START ==========================
// ❗Responsive nav & floating-controls❗
const menu = doc.getElementById("menu");

function toggleMenu() {
  setTimeout(() => {
    nav.classList.toggle("translate-nav");
    fControls.classList.toggle("translate-floating-controls");
    body.classList.toggle("blur");
    body.classList.toggle("overflow-toggle");
  }, 100);
}

// Open menu on click
menu.addEventListener("click", () => {
  toggleMenu();
});

// Close menu on click outside nav
doc.addEventListener("click", (e) => {
  if (e.target && e.target.classList.contains("blur")) {
    toggleMenu();
  } else if (e.target && e.target.id == "model-viewer-id") {
    if (nav.classList.contains("translate-nav")) {
      toggleMenu(); // Close nav
    }
  }
});
// ========================== NAV END ==========================

// ========================== SIDE START ==========================
// ❗Hide/Show info❗
function toggleInfo() {
  setTimeout(() => {
    body.style.setProperty(
      "--side-current-width",
      side.clientWidth + 20 + "px"
    );
    side.classList.toggle("hide-side");
    modelArea.classList.toggle("hide-side-model-area");
  }, 100);

  infoBtnText = doc.querySelectorAll("#info-btn > span")[1].textContent;
  if (infoBtnText == "Show information") {
    doc.querySelectorAll("#info-btn > span")[1].textContent =
      "Hide information";
  } else {
    doc.querySelectorAll("#info-btn > span")[1].textContent =
      "Show information";
  }
}

infoBtn.addEventListener("click", toggleInfo);
// ========================== SIDE END ==========================

// ========================== FORMAT DATA-LIST START ==========================
// ❗Format data-list❗
var options = $("#meshes option")
  .map(function () {
    return this.value;
  })
  .get();

var mesh_names = options.map(function (val) {
  // Replace '__' with ' / ' and '_' with ' '
  return val.split("__").join(" / ").replace("_", " ");
});

$("#meshes option").each(function (i) {
  $(this).val(mesh_names[i]);
});
// ========================== FORMAT DATA-LIST END ==========================

// ========================== AJAX SEARCH START ==========================
// ❗Handle search❗
$("#search").on("input", function (e) {
  // LATE_EXP: Marked for refactor + views.py.
  e.preventDefault();

  $.ajax({
    type: "GET",
    url: PRE_URL + "/search/",
    data: { query: $(this).val() },
    dataType: "json",
    success: function (resp) {
      if (resp.meshes_json != null) {
        $(".models").empty(); // Clear list
        $.each(resp.meshes_json, function (mesh) {
          $(".models").append(
            "<a class='model-previews' href='" +
              PRE_URL +
              "/models/" +
              resp.meshes_json[mesh].verbose_id +
              "/'><div class='model-status' style='background-color: " +
              resp.meshes_json[mesh].completed_col +
              ";'>" +
              resp.meshes_json[mesh].completed_msg +
              "</div>" +
              "<div class='model-preview-inner' style='background-image: url(" +
              resp.meshes_json[mesh].thumb_url +
              "); background-size: cover;'><div class='model-title'>" +
              mesh +
              "</div></div></a>"
          );
        });
      } else {
        $(".models").html("<p>No matches were found.</p>");
      }
    },
    error: function (resp) {
      console.log("GET ERROR in search.");
    },
  });
});
// ========================== AJAX SEARCH END ==========================

// ========================== AJAX RUN LOAD START ==========================
// ❗Handle run load❗
$("#select-run").on("change", function (e) {
  e.preventDefault();
  var runark = $(this).val();
  var page_vid = window.location.pathname.split("/")[4];

  $.ajax({
    type: "GET",
    url: PRE_URL + "/loadRun/",
    data: { runark: runark },
    success: function (resp) {
      if (resp.run != null) {
        // NOTE: Fix for the apparent model-viewer memory leak
        // LATE_EXP: TODO: Expose this as an experimental setting in the UI for users to toggle.
        customElements.get("model-viewer").modelCacheSize = 0;

        // Load mesh
        $("model-viewer").attr("src", resp.run.mesh_src);
        $("model-viewer").attr("orientation", resp.run.orientation);
        // Run details
        $("#latest-recons").html("Reconstructed on: " + resp.run.ended_at);
        $("#contrib-count").html("Contributors: " + resp.run.contrib_count);
        $("#images-count").html("Images: " + resp.run.images_count);
        $("#run-ark-link").html(resp.run.run_ark);
        $("#run-ark-link").attr("href", resp.run.run_ark_url);

        // Change page title
        $(doc).attr("title", "Project Tirtha | " + resp.run.mesh_name);
        // Change page url
        window.history.pushState(
          "",
          "",
          window.location.origin +
            PRE_URL +
            "/models/" +
            page_vid +
            "/" +
            resp.run.runid +
            "/"
        );
      } else {
        console.log("No matching runs were found.");
      }
    },
    error: function (resp) {
      console.log("GET ERROR in loadRun.");
    },
  });
});
// ========================== AJAX RUN LOAD END ==========================

// ========================== AJAX MESH LOAD START ==========================
// ❗Handle mesh load❗
$(".model-previews").on("click", function (e) {
  e.preventDefault();
  var vid = $(this).attr("href").split("/")[4];
  var page_vid = window.location.pathname.split("/")[3];

  var modStat = $(this).find(".model-status");

  $.ajax({
    type: "GET",
    url: PRE_URL + "/loadMesh/",
    data: { vid: vid },
    success: function (resp) {
      if (resp.mesh != null) {
        if (resp.mesh.has_run == false) {
          const ms_html = modStat.html();
          const ms_bg = modStat.css("background-color");

          // Set model-status to this for 5 seconds
          modStat.html("Model pending");
          modStat.css("background-color", "orange");

          // Change model-status back after 5 seconds
          setTimeout(function () {
            modStat.html(ms_html);
            modStat.css("background-color", ms_bg);
          }, 5000);
        } else {
          // NOTE: Fix for the apparent model-viewer memory leak
          customElements.get("model-viewer").modelCacheSize = 0;

          $("model-viewer").attr("poster", resp.mesh.prev_url);
          $("model-viewer").attr("src", resp.mesh.src);
          $("model-viewer").attr("orientation", resp.mesh.orientation);

          $("#mesh-name").html("About " + resp.mesh.name);
          $("#mesh-info").html(resp.mesh.desc);
          // Last reconstructed time
          $("#latest-recons").html(
            "Reconstructed on: " + resp.mesh.last_recons
          );
          // Contrib stat
          if (resp.mesh.contrib_type == "run")
            $("#contrib-count").html(
              "Contributors: " + resp.mesh.contrib_count
            );
          else
            $("#contrib-count").html(
              "Contributions: " + resp.mesh.contrib_count
            );
          $("#images-count").html("Images: " + resp.mesh.images_count);
          // ARK
          $("#run-ark-link").html(resp.mesh.run_ark);
          $("#run-ark-link").attr("href", resp.mesh.run_ark_url);
          // Populate #select-run
          $("#select-run").html("");
          $.each(resp.mesh.runs_arks, function (idx, runark) {
            $("#select-run").append(
              "<option value='" + runark + "'>" + runark + "</option>"
            );
          });

          // Change page title & url
          $(doc).attr("title", "Project Tirtha | " + resp.mesh.name);
          if (vid != page_vid) {
            window.history.pushState(
              "",
              "",
              window.location.origin + PRE_URL + "/models/" + vid
            );
          }
        }
      } else {
        console.log("No matching meshes were found.");
      }
    },
    error: function (resp) {
      console.log("GET ERROR in loadMesh.");
    },
  });
});
// ========================== AJAX MESH LOAD END ==========================

// ========================== HQ DOWNLOAD START ==========================
// TODO:❗Handle HQ download❗ - LATE_EXP: For later.
// $('#hq').click(function() {
//     // Add download functionality
//     // TODO: options for "data products"
//     // downloadDialog.showModal();
//     // body.classList.toggle("overflow-toggle");
//     alert("Coming soon! Stay tuned.")
// });
// ========================== HQ DOWNLOAD END ==========================

// ========================== MODAL START ==========================
// ❗Handle contribute modal❗
const upInput = doc.getElementById("upload-input");
const contBtn = doc.getElementById("cont-btn");
const clearBtn = doc.getElementById("clear-btn");
const subBtn = doc.getElementById("submit-btn");
const upLabel = doc.getElementById("upload-label");
const contDialog = doc.getElementById("cont-form");
var upGal = doc.getElementById("upload-gallery");
var compressedFiles = [];
const selectedFiles = new Set();
const MAX_FILES = 1500, // NOTE: Limit to 1500 images. Tweak as needed.
  THUMB_DIM = 300, // Thumbnail dimensions
  MIN_DIM = 1080, // Minimum dimensions for filtering
  UPPER_DIM = 2160, // Upper limit, used for scaling down
  SCALE = 2, // Scale factor for resizing (1 / SCALE)
  QUAL = 0.7; // JPEG quality

async function checkExif(file) {
  return new Promise((resolve, reject) => {
    var reader = new FileReader();
    reader.onload = function (e) {
      var resultArray = new Uint8Array(e.target.result);
      var exifMarker = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00]; // "Exif\0\0"

      // Search for the Exif marker
      for (var i = 0; i < resultArray.length - exifMarker.length; i++) {
        var found = true;
        for (var j = 0; j < exifMarker.length; j++) {
          if (resultArray[i + j] !== exifMarker[j]) {
            found = false;
            break;
          }
        }
        if (found) resolve(true);
      }
      resolve(false);
    };
    reader.readAsArrayBuffer(file);
  });
}

// Show modal with SO & Form
contBtn.addEventListener("click", function () {
  contDialog.showModal();
  body.classList.toggle("overflow-toggle");
});

// Close modal on receiving a click outside
contDialog.addEventListener("click", function (el) {
  if (el.target.id == "cont-form") {
    body.classList.toggle("overflow-toggle");
    contDialog.close();
  }
});

// Adds files to upInput using DataTransfer
function syncInput() {
  const dt = new DataTransfer();
  selectedFiles.forEach(function (file) {
    dt.items.add(file);
  });
  upInput.files = dt.files;
}

// Update counter
function updateCounter() {
  if (selectedFiles.size == 0) $("#file-count").html("No files selected.");
  else
    $("#file-count").html(
      "Validated & compressed " +
        selectedFiles.size +
        " / " +
        MAX_FILES +
        " files."
    );
}

// Adds files to selectedFiles & syncs with upInput
function addFile(file) {
  selectedFiles.add(file);
  syncInput();
  updateCounter();
}

// Removes file from selectedFiles & upInput
function removeFile(pic) {
  const name = pic.name;
  selectedFiles.delete(pic);
  // Match by name and delete from compressedFiles
  compressedFiles = compressedFiles.filter(function (el) {
    return el.name !== name;
  });

  syncInput();
  updateCounter();
}

// Aborts the async function from another function
var controller;

function abortGallery() {
  controller.abort();
}

// Converts dataURL to blob
const dataURLToBlob = (dataURL) => {
  const arr = dataURL.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
};

function formChange() {
  upLabel.classList.remove("loading");
  subBtn.disabled = false;
  subBtn.classList.remove("disabled-btn");
  clearBtn.disabled = false;
  clearBtn.classList.remove("disabled-btn");
}

// Creates image gallery
async function createGallery(files, signal) {
  try {
    // Add loading animation & disable buttons
    upLabel.classList.add("loading");
    subBtn.disabled = true;
    subBtn.classList.add("disabled-btn");
    clearBtn.disabled = true;
    clearBtn.classList.add("disabled-btn");

    const fragment = doc.createDocumentFragment();
    const createElem = doc.createElement.bind(doc);

    for (let i = 0; i < files.length; i++) {
      const pic = files[i];

      // Add pic to gallery only if it exists in selectedFiles, post validation
      if (selectedFiles.size < MAX_FILES) {
        // Validate filetype
        if (!pic.type.startsWith("image/")) {
          alert(
            `File ${pic.name} is not a supported filetype! It will be ignored.`
          );
          continue;
        }

        // Check if Exif data is present | Validate Exif
        const exifResult = await checkExif(pic);
        if (!exifResult) {
          alert(`File ${pic.name} has no Exif data. It will be ignored.`);
          continue;
        }

        // Add <img> element
        const reader = new FileReader();
        await new Promise((resolve, reject) => {
          reader.onloadend = function () {
            // Check if `Clear` was clicked
            if (signal.aborted) return;

            // Compress + resize image
            const image = new Image();
            image.src = reader.result;
            var add = true;
            image.onload = async () => {
              let a = createElem("a");

              let imgWidth = image.width;
              let imgHeight = image.height;

              // Validate resolution
              if (imgWidth < MIN_DIM || imgHeight < MIN_DIM) {
                alert(
                  `File ${pic.name} has a low resolution! It will be ignored.`
                );
                add = false;
              }

              // All validations passed -> Add to selectedFiles & upInput & update counter
              if (add) {
                addFile(pic);

                // Resize
                if (imgWidth > UPPER_DIM || imgHeight > UPPER_DIM) {
                  imgWidth = Math.round(imgWidth / SCALE);
                  imgHeight = Math.round(imgHeight / SCALE);
                }
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                canvas.width = imgWidth;
                canvas.height = imgHeight;
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

                // Compress
                const dataURL = canvas.toDataURL("image/jpeg", QUAL);

                // Copy EXIF data
                let blob = dataURLToBlob(dataURL);
                // TODO: LATE_EXP: Add test to confirm that checkExif() and copyExif() agree
                const newImage = new File(
                  [await copyExif(pic, blob)],
                  pic.name,
                  {
                    type: "image/jpeg",
                  }
                );
                compressedFiles.push(newImage);

                // Edit <a> element
                a.href = URL.createObjectURL(newImage);
                a.target = "_blank";

                // Create thumbnail
                let thumbCanvas = document.createElement("canvas");
                let THUMB_DIM_LOC = Math.min(THUMB_DIM, imgWidth, imgHeight);
                thumbCanvas.width = THUMB_DIM_LOC;
                thumbCanvas.height = THUMB_DIM_LOC;
                let thumbCtx = thumbCanvas.getContext("2d");
                thumbCtx.drawImage(image, 0, 0, THUMB_DIM_LOC, THUMB_DIM_LOC);
                thumbCanvas.toBlob((blob) => {
                  // Add <img> element with thumbnail
                  const thumbImg = createElem("img");
                  thumbImg.src = URL.createObjectURL(blob);
                  thumbImg.classList.add("added-pic");
                  a.appendChild(thumbImg);
                }, "image/jpeg");

                // Add remove button
                let removeBtn = createElem("button");
                removeBtn.innerHTML = "✕";
                removeBtn.classList.add("remove");
                removeBtn.addEventListener("click", function (e) {
                  e.preventDefault();
                  this.parentElement.remove();
                  removeFile(pic);
                });
                a.appendChild(removeBtn);

                // Add <a> to fragment
                fragment.appendChild(a);
              }
              resolve();
            };
          };
          reader.readAsDataURL(pic);
        });
      }
    }
    // Remove loading animation & enable buttons
    formChange();

    // Add fragment to gallery
    upGal.appendChild(fragment);
  } catch (err) {
    console.error("Gallery Error: " + err);
  }
}

upInput.addEventListener(
  "change",
  function (e) {
    e.preventDefault();
    let ctrl = new AbortController();
    controller = ctrl;
    createGallery(this.files, ctrl.signal);
  },
  false
);

// Handles gallery clear
function clearGallery() {
  abortGallery();
  const images = upGal.getElementsByTagName("a");
  while (images.length > 0) upGal.removeChild(images[0]);
  upInput.value = "";
  selectedFiles.clear();
  compressedFiles = [];
  $("progress").val(0);
}

clearBtn.addEventListener(
  "click",
  function () {
    if (selectedFiles.size > 0 && compressedFiles.length > 0) {
      clearGallery();
      $("#file-count").html("No files selected.");
    }
  },
  false
);
// ========================== DRAG & DROP START ==========================
// ❗Handle drag & drop❗ | Adapted from: https://codepen.io/joezimjs/pen/yPWQbd
function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Prevents default drag behaviors
["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
  upGal.addEventListener(evt, preventDefaults, false);
  doc.body.addEventListener(evt, preventDefaults, false);
});

// Highlights drop area, when item is dragged over it
["dragenter", "dragover"].forEach((evt) => {
  upGal.addEventListener(
    evt,
    () => {
      upGal.classList.add("highlight");
    },
    false
  );
});
["dragleave", "drop"].forEach((evt) => {
  upGal.addEventListener(
    evt,
    () => {
      upGal.classList.remove("highlight");
    },
    false
  );
});

// Handles dragged & dropped files
upGal.addEventListener(
  "drop",
  function (e) {
    let ctrl = new AbortController();
    controller = ctrl;
    createGallery(e.dataTransfer.files, ctrl.signal);
  },
  false
);
// ========================== DRAG & DROP END ==========================

// ========================== AJAX UPLOAD START ==========================
uploadForm.on("submit", function (e) {
  e.preventDefault();

  // Sync compressedFiles with upInput
  const dt = new DataTransfer();
  compressedFiles.forEach(function (file) {
    dt.items.add(file);
  });
  upInput.files = dt.files;

  // 0-files workaround for Safari: Check if selected files has non-zero length
  if (compressedFiles.length == 0) {
    $("#file-count").html("No files selected.");
    // Set text-shadow to draw attention for 1s
    $("#file-count").css("text-shadow", "0 0 10px #ff0000");
    setTimeout(function () {
      $("#file-count").css("text-shadow", "none");
    }, 1000);

    return;
  }

  // Disable clear button
  clearBtn.disabled = true;
  clearBtn.classList.add("disabled-btn");

  var formData = new FormData(uploadForm[0]);
  // Convert mesh_name to verbose_id - replace ' / ' with '__' & ' ' with '_'
  var mesh_vid = formData.get("mesh").split(" / ").join("__").replace(" ", "_");
  formData.append("mesh_vid", mesh_vid);
  formData.delete("mesh");

  // Check if the mesh_vid is valid and if so, whether the mesh is "completed"
  $("#upload-result").html("Please wait! Checking...");
  $.ajax({
    type: "GET",
    url: PRE_URL + "/preUpload/",
    data: { mesh_vid: mesh_vid },
    success: function (resp) {
      $("#upload-result").html(resp.output);
      if (resp.allowupload == true) uploadFiles();
      else {
        clearBtn.disabled = false;
        clearBtn.classList.remove("disabled-btn");
        if (resp.blur == true) {
          uploadForm.addClass("blur-form");
          uploadFormElems.attr("inert", "");
        }
      }
    },
    error: function (resp) {
      console.log("GET ERROR in preUpload.");
      clearBtn.disabled = false;
      clearBtn.classList.remove("disabled-btn");
      $("#upload-result").html("Error! Please try again.");
    },
  });

  function uploadFiles() {
    // Upload files
    $("#upload-result").html("Please wait! Uploading...");
    $.ajax({
      type: "POST",
      url: PRE_URL + "/upload/",
      data: formData,
      dataType: "json",
      processData: false,
      contentType: false,
      xhr: function () {
        var xhr = new window.XMLHttpRequest();
        xhr.upload.addEventListener(
          "progress",
          function (e) {
            if (e.lengthComputable) {
              var percentComplete = (e.loaded / e.total) * 100;
              $("progress").val(percentComplete);
            }
          },
          false
        );
        return xhr;
      },
      success: function (resp) {
        if (resp["status"] == "Success") {
          uploadForm.trigger("reset");
          clearBtn.disabled = false;
          clearBtn.classList.remove("disabled-btn");
          clearGallery();
          $("#file-count").html("No files selected.");
        }
        $("#select-country").focus();
        $("#upload-result").html(resp.output);
      },
      error: function (resp) {
        console.log("POST ERROR in upload.");
        clearBtn.disabled = false;
        clearBtn.classList.remove("disabled-btn");
        $("#upload-result").html("Error! Please try again.");
      },
    });
  }
});
// ========================== AJAX UPLOAD END ==========================
// ========================== MODAL END ==========================
