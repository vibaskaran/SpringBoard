
var createFilters = function(){
  // //CHECKBOX FUNCTIONS
  // //When user clicks on a filter, only show selected information
  var input_types = [
    'education',
    'technology',
    'travel',
    'science',
    'music'

  ];

  //Making the inputs
  var ul = d3.select("ul#input-types")
    .selectAll('li')
    .data(input_types)
    .enter()
    .append('li')
    .append('p')
    .text(function(d){
      return d;
  });
  //making them into checkboxes
  ul.append('input')
    .attr('type', "checkbox")
    .attr('id', function(d){
      return d;
    });



  d3.select('#education').on('change', update);
  update();

  function update(){
    if(d3.select('#education').property('checked') == true){
      console.log('checked');
      console.log(data);

    }else{
      console.log('unchecked')
    }
  }

}


//===============
//global function for data
// var dataset;

// var clickData = d3.csv("assets/data/ted_w_images.csv", function(data) {
//   dataset = data;
// });


// setTimeout(function(){
//   d3.select('#education').on('change', update);
//   update();

//   function update(){
//     if(d3.select('#education').property('checked') == true){
//       console.log('checked');
//       console.log(dataset);

//     }else{
//       console.log('unchecked')
//     }
//   }

// },2000);



// D3 Animated Scatter Plot

// Section 1: Pre-Data Setup
// ===========================
// Before we code any data visualizations,
// we need to at least set up the width, height and margins of the graph.
// Note: I also added room for label text as well as text padding,
// though not all graphs will need those specifications.

// Grab the width of the containing box
var width = parseInt(d3.select("#scatter").style("width"));

// Designate the height of the graph
var height = width - width / 3.9;

// Margin spacing for graph
var margin = 20;

// space for placing words
var labelArea = 110;

// padding for the text at the bottom and left axes
var tPadBot = 40;
var tPadLeft = 40;

// Create the actual canvas for the graph
var svg = d3
  .select("#scatter")
  .append("svg")
  .attr("width", width)
  .attr("height", height)
  .attr("class", "chart");



// Set the radius for each dot that will appear in the graph.
// Note: Making this a function allows us to easily call
// it in the mobility section of our code.
var circRadius;
function crGet() {
  if (width <= 530) {
    circRadius = 5;
  }
  else {
    circRadius = 15;
  }
}
crGet();

// var parser = d3.timeParse("%d/%m/%Y");

// The Labels for our Axes

// A) Bottom Axis
// ==============

// We create a group element to nest our bottom axes labels.
svg.append("g").attr("class", "xText");
// xText will allows us to select the group without excess code.
var xText = d3.select(".xText");

// We give xText a transform property that places it at the bottom of the chart.
// By nesting this attribute in a function, we can easily change the location of the label group
// whenever the width of the window changes.
function xTextRefresh() {
  xText.attr(
    "transform",
    "translate(" +
      ((width - labelArea) / 2.5 + labelArea) +
      ", " +
      (height - margin - tPadBot) +
      ")"
  );
}
xTextRefresh();

// Now we use xText to append three text SVGs, with y coordinates specified to space out the values.
// 1. Poverty
xText
  .append("text")
  .attr("y", -26)
  .attr("data-name", "comments")
  .attr("data-axis", "x")
  .attr("class", "aText active x")
  .text("Comments");
// // 2. Age
<<<<<<< HEAD:assets/js/app_org.js
// xText
//   .append("text")
//   .attr("y", 0)
//   .attr("data-name", "comments")
//   .attr("data-axis", "x")
//   .attr("class", "aText inactive x")
//   .text("comments");
// // 3. Income
// xText
//   .append("text")
//   .attr("y", 26)
//   .attr("data-name", "income")
//   .attr("data-axis", "x")
//   .attr("class", "aText inactive x")
//   .text("Household Income (Median)");
=======
xText
  .append("text")
  .attr("y", 0)
  .attr("data-name", "film_date")
  .attr("data-axis", "x")
  .attr("class", "aText inactive x")
  .text("Years");
// 3. Income
xText
  .append("text")
  .attr("y", 26)
  .attr("data-name", "views")
  .attr("data-axis", "x")
  .attr("class", "aText inactive x")
  .text("View Count");
>>>>>>> master:D3Chart/assets/js/app.js

// B) Left Axis
// ============

// Specifying the variables like this allows us to make our transform attributes more readable.
var leftTextX = margin + tPadLeft;
var leftTextY = (height + labelArea) / 2 - labelArea;

// We add a second label group, this time for the axis left of the chart.
svg.append("g").attr("class", "yText");

// yText will allows us to select the group without excess code.
var yText = d3.select(".yText");

// Like before, we nest the group's transform attr in a function
// to make changing it on window change an easy operation.
function yTextRefresh() {
  yText.attr(
    "transform",
    "translate(" + leftTextX + ", " + leftTextY + ")rotate(-90)"
  );
}
yTextRefresh();

// Now we append the text.
// 1. views
yText
  .append("text")
  .attr("y", -26)
  .attr("data-name", "views")
  .attr("data-axis", "y")
  .attr("class", "aText active y")
  .text("View Count");

<<<<<<< HEAD:assets/js/app_org.js
// 2. Smokes
// yText
//   .append("text")
//   .attr("x", 0)
//   .attr("data-name", "comments")
//   .attr("data-axis", "y")
//   .attr("class", "aText inactive y")
//   .text("comments");

// // 3. Lacks Healthcare
// yText
//   .append("text")
//   .attr("y", 26)
//   .attr("data-name", "healthcare")
//   .attr("data-axis", "y")
//   .attr("class", "aText inactive y")
//   .text("Lacks Healthcare (%)");
=======
// 2. Comments
yText
  .append("text")
  .attr("x", 0)
  .attr("data-name", "comments")
  .attr("data-axis", "y")
  .attr("class", "aText inactive y")
  .text("Comments");

// 3. Ratings
yText
  .append("text")
  .attr("y", 26)
  .attr("data-name", "ratings")
  .attr("data-axis", "y")
  .attr("class", "aText inactive y")
  .text("Ratings");
>>>>>>> master:D3Chart/assets/js/app.js

// 2. Import our .csv file.
// ========================
// This data file ted talk information with images

// Import our CSV data with d3's .csv import method.
<<<<<<< HEAD:assets/js/app_org.js
d=d3.csv("assets/data/ted_main_clean.csv", function(data) {
=======
d = d3.csv("assets/data/ted_w_images.csv", function(data) {
>>>>>>> master:D3Chart/assets/js/app.js
  // Visualize the data
  visualize(data);
  createFilters();
});


// 3. Create our visualization function
// ====================================
// We called a "visualize" function on the data obtained with d3's .csv method.
// This function handles the visual manipulation of all elements dependent on the data.
function visualize(theData) {


  // PART 1: Essential Local Variables and Functions
  // =================================
  // curX and curY will determine what data get's represented in each axis.
  // We designate our defaults here, which carry the same names
  // as the headings in their matching .csv data file.
  var curX = "comments";
  var curY = "views";

  // We also save empty variables for our the min and max values of x and y.
  // this will allow us to alter the values in functions and remove repititious code.
  var xMin;
  var xMax;
  var yMin;
  var yMax;

  // This function allows us to set up tooltip rules (see d3-tip.js).
  var toolTip = d3
    .tip()
    .attr("class", "d3-tip")
    .offset([40, -60])
    .html(function(d) {
      // x key
      var theX;
      // Grab the state name.
      var theTitle = "<div>" + d.title + "</div>";
      var theSpeaker = "<div>Main Speaker:" + d.main_speaker + "</div>";
      // var theFilmDate = "<div>" + "Filmed Date: " + d.film_date + "</div>";
      // Snatch the y value's key and value.
      var theY = "<div>" + curY + ": " + d[curY] + "</div>";
<<<<<<< HEAD:assets/js/app_org.js
      // If the x key is poverty
      if (curX === "views") {
        // Grab the x key and a version of the value formatted to show percentage
        theX = "<div>" + curX + ": " + d[curX] + "</div>";
      }
      else {
=======

      //MARIAH: EXAMPLE OF PUTTING A PICTURE INSIDE THE TOOLTIP
      var thePicture = "<img src='" + d.image_url + "' style='width:70px;height:70px'>"
      

      //MARIAH: SETTING THE X VALUES
      if (curX === "views") {
        // Grab the x key and a version of the value formatted to show percentage
        theX = "<div>" + curX + ": " + d[curX] + "</div>";
      }else if(curX === "comments"){
>>>>>>> master:D3Chart/assets/js/app.js
        // Otherwise
        // Grab the x key and a version of the value formatted to include commas after every third digit.
        theX = "<div>" +
          curX +
          ": " +
          parseFloat(d[curX]).toLocaleString("en") +
          "</div>";
      }else if(curX == "Years"){
        // Grab the x key and a version of the value formatted to include commas after every third digit.
        theX = "<div>" +
          curX +
          ": " +
          parseFloat(d[curX]).toLocaleString("en") +
          "</div>";
      }

      // Display what we capture.
<<<<<<< HEAD:assets/js/app_org.js
      return theTitle + theFilmDate + theX + theY;
=======
      return theSpeaker + theTitle  + theX + theY + thePicture;

>>>>>>> master:D3Chart/assets/js/app.js
    });
  // Call the toolTip function.
  svg.call(toolTip);

  // PART 2: D.R.Y!
  // ==============
  // These functions remove some repitition from later code.
  // This will be more obvious in parts 3 and 4.

  // a. change the min and max for x
  function xMinMax() {
    // min will grab the smallest datum from the selected column.
    xMin = d3.min(theData, function(d) {
      return parseFloat(d[curX]) * 0.90;
    });

    // .max will grab the largest datum from the selected column.
    xMax = d3.max(theData, function(d) {
      return parseFloat(d[curX]) * 1.10;
    });
  }

  // b. change the min and max for y
  function yMinMax() {
    // min will grab the smallest datum from the selected column.
    yMin = d3.min(theData, function(d) {
      return parseFloat(d[curY]) * 0.90;
    });

    // .max will grab the largest datum from the selected column.
    yMax = d3.max(theData, function(d) {
      return parseFloat(d[curY]) * 1.10;
    });
  }

  // c. change the classes (and appearance) of label text when clicked.
  function labelChange(axis, clickedText) {
    // Switch the currently active to inactive.
    d3
      .selectAll(".aText")
      .filter("." + axis)
      .filter(".active")
      .classed("active", false)
      .classed("inactive", true);

    // Switch the text just clicked to active.
    clickedText.classed("inactive", false).classed("active", true);
  }

  // Part 3: Instantiate the Scatter Plot
  // ====================================
  // This will add the first placement of our data and axes to the scatter plot.

  // First grab the min and max values of x and y.
  xMinMax();
  yMinMax();

  // With the min and max values now defined, we can create our scales.
  // Notice in the range method how we include the margin and word area.
  // This tells d3 to place our circles in an area starting after the margin and word area.
  var xScale = d3
    .scaleLog()//Scale logarithmically 
    .range([margin + labelArea, width - margin])
    .domain([xMin, xMax]);
    
  var yScale = d3
    .scaleLog()//Scale logarithmically 
    .domain([yMin, yMax])
    // Height is inversed due to how d3 calc's y-axis placement
    .range([height - margin - labelArea, margin]);

  // We pass the scales into the axis methods to create the axes.
  // Here we turn the ticks from the log numbers to actual numbers
  var xAxis = d3.axisBottom(xScale).tickFormat(function (d) {
        return xScale.tickFormat(4,d3.format(",d"))(d)
  });
  var yAxis = d3.axisLeft(yScale).tickFormat(function (d) {
        return yScale.tickFormat(4,d3.format(",d"))(d)
  });

  // Determine x and y tick counts.
  // Note: Saved as a function for easy mobile updates.
  function tickCount() {
    if (width <= 500) {
      xAxis.ticks(5);
      yAxis.ticks(5);
    }
    else {
      xAxis.ticks(10);
      yAxis.ticks(10);
    }
  }
  tickCount();

  // We append the axes in group elements. By calling them, we include
  // all of the numbers, borders and ticks.
  // The transform attribute specifies where to place the axes.
  svg
    .append("g")
    .call(xAxis)
    .attr("class", "xAxis")
    .attr("transform", "translate(0," + (height - margin - labelArea) + ")");
  svg
    .append("g")
    .call(yAxis)
    .attr("class", "yAxis")
    .attr("transform", "translate(" + (margin + labelArea) + ", 0)");

  var config = {
<<<<<<< HEAD:assets/js/app_org.js
    "avatar_size" : 35
	}
  
  var defs = svg.append('svg:defs');

  defs.append("svg:pattern")
    // .attr("id", function(d) { return (d.title+"-icon-img");})
    .attr("id", "ted_icon")
    .attr("width", config.avatar_size)
    .attr("height", config.avatar_size)
    .attr("patternUnits", "userSpaceOnUse")
    .append("svg:image")
    // .attr("xlink:xlink:href", function(d) { return (d.url);})
    // .attr("xlink:xlink:href", 'https://cdnarchitect.s3.amazonaws.com/wp-content/uploads/2017/06/TG-11-TEDTalks.jpg')
    .attr("xlink:xlink:href", 'https://pi.tedcdn.com/r/pe.tedcdn.com/images/ted/7c1c78e12cf3c1134ff629675bee896bfebabfec_254x191.jpg?')
    .attr("width", config.avatar_size)
    .attr("height", config.avatar_size)
    .attr("x", 0)
    .attr("y", 0)
    .attr("preserveAspectRatio", "xMinYMin slice");
    
=======
    "avatar_size" : 20,
	}

    var defs = svg.append('svg:defs');

    defs.append("svg:pattern")
      .attr("id", "ted_icon")
      .attr("width", config.avatar_size)
      .attr("height", config.avatar_size)
      .attr("patternUnits", "objectBoundingBox")
      .append("svg:image")
        .attr("xlink:xlink:href", '../icon.jpg')
        .attr("width", config.avatar_size + 20)
        .attr("height", config.avatar_size + 20)
        .attr("x", 0)
        .attr("y", 0)
        .attr("preserveAspectRatio", "xMinYMin slice");


>>>>>>> master:D3Chart/assets/js/app.js
  // Now let's make a grouping for our dots and their labels.
  var theCircles = svg.selectAll("g theCircles").data(theData).enter();
  

  // We append the circles for each row of data.
  theCircles
    .append("circle")
    // These attr's specify location, size and class.
    .attr("cx", function(d) {
      return xScale(d[curX]);
    })
    .attr("cy", function(d) {
      return yScale(d[curY]);
    })
    .attr("r", circRadius)    
    .style("fill", "#fff")
    // .style("fill", "url("+d.title+"-icon-img)")
    .style("fill", "url(#ted_icon)")
    .attr("class", function(d) {
      return "stateCircle " + d.main_speaker
    })
    // Hover rules
    .on("mouseover", function(d) {
      // Show the tooltip
      toolTip.show(d);
      // Highlight the state circle's border
      d3.select(this).style("stroke", "#323232");
    })
    .on("mouseout", function(d) {
      // Remove the tooltip
      toolTip.hide(d);
      // Remove highlight
      d3.select(this).style("stroke", "#e3e3e3");
    });

  // With the circles on our graph, we need matching labels.
  // Let's grab the state abbreviations from our data
  // and place them in the center of our dots.
  theCircles
    .append("text")
    // We return the abbreviation to .text, which makes the text the abbreviation.
    .text(function(d) {
      return d.main_speaker;
    })
    // Now place the text using our scale.
    .attr("dx", function(d) {
      return xScale(d[curX]);
    })
    .attr("dy", function(d) {
      // When the size of the text is the radius,
      // adding a third of the radius to the height
      // pushes it into the middle of the circle.
      return yScale(d[curY]) + circRadius / 2.5;
    })
    .attr("font-size", circRadius/3)
    .attr("class", "stateText")
    // Hover Rules
    .on("mouseover", function(d) {
      // Show the tooltip
      toolTip.show(d);
      // Highlight the state circle's border
      d3.select("." + d.title).style("stroke", "#323232");
    })
    .on("mouseout", function(d) {
      // Remove tooltip
      toolTip.hide(d);
      // Remove highlight
      d3.select("." + d.title).style("stroke", "#e3e3e3");
    });



  // Part 4: Dynamize the Graph
  // ==========================
  // This section will allow the user to click on any label
  // and display the data it references.

  // Select all axis text and add this d3 click event.
  d3.selectAll(".aText").on("click", function() {
    // Make sure we save a selection of the clicked text,
    // so we can reference it without typing out the invoker each time.
    var self = d3.select(this);

    // We only want to run this on inactive labels.
    // It's a waste of the processor to execute the function
    // if the data is already displayed on the graph.
    if (self.classed("inactive")) {
      // Grab the name and axis saved in label.
      var axis = self.attr("data-axis");
      var name = self.attr("data-name");

      // When x is the saved axis, execute this:
      if (axis === "x") {
        // Make curX the same as the data name.
        curX = name;

        // Change the min and max of the x-axis
        xMinMax();

        // Update the domain of x.
        xScale.domain([xMin, xMax]);

        // Now use a transition when we update the xAxis.
        svg.select(".xAxis").transition().duration(1000).call(xAxis);

        // With the axis changed, let's update the location of the state circles.
        d3.selectAll("circle").each(function() {
          // Each state circle gets a transition for it's new attribute.
          // This will lend the circle a motion tween
          // from it's original spot to the new location.
          d3
            .select(this)
            .transition()
            .attr("cx", function(d) {
              return xScale(d[curX]);
            })
            .duration(1800);
        });

        // We need change the location of the state texts, too.
        d3.selectAll(".stateText").each(function() {
          // We give each state text the same motion tween as the matching circle.
          d3
            .select(this)
            .transition()
            .attr("dx", function(d) {
              return xScale(d[curX]);
            })
            .duration(1800);
        });

        // Finally, change the classes of the last active label and the clicked label.
        labelChange(axis, self);
      }
      else {
        // When y is the saved axis, execute this:
        // Make curY the same as the data name.
        curY = name;

        // Change the min and max of the y-axis.
        yMinMax();

        // Update the domain of y.
        yScale.domain([yMin, yMax]);

        // Update Y Axis
        svg.select(".yAxis").transition().duration(1000).call(yAxis);

        // With the axis changed, let's update the location of the state circles.
        d3.selectAll("circle").each(function() {
          // Each state circle gets a transition for it's new attribute.
          // This will lend the circle a motion tween
          // from it's original spot to the new location.
          d3
            .select(this)
            .transition()
            .attr("cy", function(d) {
              return yScale(d[curY]);
            })
            .duration(1800);
        });

        // We need change the location of the state texts, too.
        d3.selectAll(".stateText").each(function() {
          // We give each state text the same motion tween as the matching circle.
          d3
            .select(this)
            .transition()
            .attr("dy", function(d) {
              return yScale(d[curY]) + circRadius / 3;
            })
            .duration(1800);
        });

        // Finally, change the classes of the last active label and the clicked label.
        labelChange(axis, self);
      }
    }
  });

  // Part 5: Mobile Responsive
  // =========================
  // With d3, we can call a resize function whenever the window dimensions change.
  // This make's it possible to add true mobile-responsiveness to our charts.
  d3.select(window).on("resize", resize);

  // One caveat: we need to specify what specific parts of the chart need size and position changes.
  function resize() {
    // Redefine the width, height and leftTextY (the three variables dependent on the width of the window).
    width = parseInt(d3.select("#scatter").style("width"));
    height = width - width / 3.9;
    leftTextY = (height + labelArea) / 2 - labelArea;

    // Apply the width and height to the svg canvas.
    svg.attr("width", width).attr("height", height);

    // Change the xScale and yScale ranges
    xScale.range([margin + labelArea, width - margin]);
    yScale.range([height - margin - labelArea, margin]);

    // With the scales changes, update the axes (and the height of the x-axis)
    svg
      .select(".xAxis")
      .call(xAxis)
      .attr("transform", "translate(0," + (height - margin - labelArea) + ")");

    svg.select(".yAxis").call(yAxis);

    // Update the ticks on each axis.
    tickCount();

    // Update the labels.
    xTextRefresh();
    yTextRefresh();

    // Update the radius of each dot.
    crGet();

    // With the axis changed, let's update the location and radius of the state circles.
    d3
      .selectAll("circle")
      .attr("cy", function(d) {
        return yScale(d[curY]);
      })
      .attr("cx", function(d) {
        return xScale(d[curX]);
      })
      .attr("r", function() {
        return circRadius;
      });

    // We need change the location and size of the state texts, too.
    d3
      .selectAll(".stateText")
      .attr("dy", function(d) {
        return yScale(d[curY]) + circRadius / 3;
      })
      .attr("dx", function(d) {
        return xScale(d[curX]);
      })
      .attr("r", circRadius / 3);
  }




}// end visualize




// console.log('asdfs')
