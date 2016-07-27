    $(".source li").draggable({
      addClasses: false,
      appendTo: "body",
      helper: "clone"
    });
     
    $(".target").droppable({
      addClasses: false,
      activeClass: "listActive",
      //accept: ":not(.ui-sortable-helper)",
      drop: function(event, ui) {
        $(this).find(".placeholder").remove();
        var link = $("<a href='#' class='dismiss'>x</a>");
        var list = $("<li></li>").text(ui.draggable.text());
        $(list).append(link);
        $(list).appendTo(this);
        //updateValues();
      }
    });
    //.sortable({
    //  items: "li:not(.placeholder)",
    //  sort: function() {
    //    $(this).removeClass("listActive");
    //  },
    //  update: function() {
    //    updateValues();
    //  }
   // }).on("click", ".dismiss", function(event) {
   //   event.preventDefault();
   //   $(this).parent().remove();
      //updateValues();
  //  });
