show_shiny <- function(){
  x1 <- seq(0,2*pi, length.out = 100)
  y1 <- sin(x1) + runif(length(x1),min=-0.1,max=0.1)
  y2 <- sin(x1)


  data1 <- tibble(x=x1,y=y1,NN=y2)
  data1 %>%
    mutate(NN=y2)


  ui <- fluidPage(
    sidebarLayout(
      sidebarPanel(
        radioButtons("art", "Art des lernens",
                     choices=list("Gradient Descend"=1,"Stochatical Gradient Descend"=2,"Random Descend"=3),selected = 2),
        uiOutput("trend_ui"),
        sliderInput("anzahl", "Anzahl hidden Layer",
                    1, 10, 3, step=1, sep=""),
        sliderInput("breite", "Breite hidden Layer",
                    1, 50, 30, step=1, sep=""),
        radioButtons("iterations", "Anzahl der Iterations",
                     choices=list("1"=1,"10"=10,"100"=100,"500"=500),
                     selected=1, inline=TRUE)
      ),
      mainPanel(
        plotlyOutput("plot")
      )))

  server <- function(input, output) {
    output$trend_ui <- renderUI({
      if (input$art==2)
        sliderInput("batches", "Anzahl der Batches",
                    1, 100, 3, step=1, sep="")
    })
    output$plot <- renderPlotly({
      a <- rep(input$breite,input$anzahl)
      print(a)
      N1 <- NN$new(length(a),c(1,a,1))
      if (input$art == 3) N1$GD3(x1,y1,delta=0.01,iteration = input$iterations)
      if (input$art == 1) for (i in 1:input$iterations) {N1$BP_reg(x1,y1,gam=0.01);print(i)}
      if (input$art == 2) N1$SGD(n = input$batches,x1,y1,delta=0.01,iteration = input$iterations)
      y3 <- N1$calculate2(x1)
      data1 %>%
        mutate(NN=y3) ->
        data1
      plt <- ggplot(data = data1, aes(x = x1))+ geom_line(aes(y=y1),show.legend = TRUE) + geom_line(aes(y=NN),color="red",show.legend = TRUE)
      ggplotly(plt)
    })
  }

  shinyApp(ui, server)
  }
